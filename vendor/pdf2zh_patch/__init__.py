"""
pdf2zh_patch — runtime patches for pdf2zh's translation engine.

Adds two behaviors the upstream engine lacks:

  PATCH#1  Text color matching.
           The original converter never emits a fill-color operator, so all
           translated text comes out black (a white title becomes black and
           unreadable). We capture each paragraph's original fill color
           (graphicstate.ncolor) and re-emit it before the translated text.

  PATCH#3  Shrink-to-fit font sizing.
           Translations (esp. zh -> en) are often longer than the source and
           overflow titles / table cells. For non-wrapping paragraphs we
           measure the translated line width and scale the font down (to at
           most 50%) so it fits the original box.

Usage: call patch() BEFORE pdf2zh.high_level builds its converter, i.e. before
running a translation. It swaps pdf2zh.converter.TranslateConverter (and the
reference high_level holds) for the patched implementation.
"""

import importlib.util
import os
import sys


def _load_patched_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "converter_patched.py")
    spec = importlib.util.spec_from_file_location("pdf2zh_patch.converter_patched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Layout classes pdf2zh normally refuses to translate (treats as graphics).
# We unlock all of them EXCEPT real math formulas, so text inside figures,
# tables, captions and "abandon" regions gets translated too. Renaming them to
# a class outside pdf2zh's protect-list (vcls) is enough — no function copy.
_UNLOCK_CLASSES = {"figure", "table", "abandon", "formula_caption"}
_KEEP_PROTECTED = {"isolate_formula"}  # never translate math formulas


def patch(translate_figures=True):
    """Replace pdf2zh's TranslateConverter with the patched one. Idempotent.

    translate_figures: when True (default), also translate text inside
    figures/tables/captions (everything except math formulas)."""
    import pdf2zh.converter as conv
    import pdf2zh.high_level as hl

    patched = _load_patched_module()

    conv.TranslateConverter = patched.TranslateConverter
    conv.Paragraph = patched.Paragraph
    conv.OpType = patched.OpType
    # high_level imports TranslateConverter by name at module load; rebind it too.
    hl.TranslateConverter = patched.TranslateConverter

    _install_color_oracle(hl)

    if translate_figures:
        _install_figure_unlock()

    return True


# Shared with the patched converter: region ids (the `i+2` values translate_patch
# assigns per detected box) that originated from a figure/table class. Inside
# those regions the converter splits text per visual line so each flowchart box /
# table cell label stays anchored at its own position instead of reflowing into
# one merged blob. Keyed by detection index i -> True.
FIGURE_BOX_INDICES = {"value": set()}


def _install_figure_unlock():
    """Wrap the layout model's predict() so protected regions (figure/table/
    caption/abandon) are re-labeled to a translatable class, AND record which
    detections are figure/table-origin so the converter can keep their labels
    in place. Math formulas (isolate_formula) stay protected."""
    import pdf2zh.doclayout as dl

    if getattr(dl, "_pdf2zh_patch_figure_unlock", False):
        return

    OnnxModel = dl.OnnxModel
    orig_predict = OnnxModel.predict

    def patched_predict(self, image, *a, **k):
        results = orig_predict(self, image, *a, **k)
        for r in results:
            names = getattr(r, "names", None)
            if not isinstance(names, dict):
                continue
            # Record figure/table-origin detection indices BEFORE renaming.
            # translate_patch enumerates r.boxes in order and assigns region id
            # = i + 2, so detection index i maps to region id i + 2.
            fig_idx = set()
            try:
                for i, d in enumerate(r.boxes):
                    if names.get(int(d.cls)) in _UNLOCK_CLASSES:
                        fig_idx.add(i + 2)  # region id used in the layout map
            except Exception:
                pass
            FIGURE_BOX_INDICES["value"] = fig_idx

            new_names = {}
            for cid, name in names.items():
                if name in _UNLOCK_CLASSES:
                    new_names[cid] = "plain text"  # not in pdf2zh's vcls -> translated
                else:
                    new_names[cid] = name
            r.names = new_names
        return results

    OnnxModel.predict = patched_predict
    dl._pdf2zh_patch_figure_unlock = True


def _build_color_oracle(doc_zh):
    """Return f(pageid, x0, y0, x1, y1)->color|None using PyMuPDF's resolved colors.

    pdfminer paragraph coords are bottom-left origin; PyMuPDF is top-left, so y
    is flipped. For a query box we collect overlapping text spans and return the
    dominant *non-black* foreground color (handles white-text-over-black-shadow,
    where pdfminer hands us the black shadow layer)."""
    # Snapshot every page's spans NOW — translate_patch wipes each page's text
    # stream before the converter runs, so a lazy read would find nothing.
    cache = {}
    try:
        for pageid in range(doc_zh.page_count):
            page = doc_zh[pageid]
            ph = page.rect.height
            spans = []
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for s in line["spans"]:
                        if not s["text"].strip():
                            continue
                        bx0, by0, bx1, by1 = s["bbox"]  # top-left origin
                        # convert to bottom-left origin to match pdfminer
                        fy0, fy1 = ph - by1, ph - by0
                        c = s["color"]
                        rgb = (((c >> 16) & 255) / 255.0,
                               ((c >> 8) & 255) / 255.0,
                               (c & 255) / 255.0)
                        spans.append((bx0, fy0, bx1, fy1, rgb))
            cache[pageid] = spans
    except Exception:
        pass

    def _spans_for_page(pageid):
        return cache.get(pageid, [])

    def _overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    def oracle(pageid, x0, y0, x1, y1):
        # Find the PyMuPDF span that best matches this query box, scored by how
        # much of the SPAN falls inside the box (so we only trust a strong
        # positional match). Return that span's ACTUAL color — including black.
        # This keeps genuinely-black text black (e.g. untranslated figure text)
        # instead of bleeding a neighbouring color onto it, while still fixing
        # the white-text-with-black-shadow case (the white span wins on overlap).
        best = None
        best_score = 0.0
        for sx0, sy0, sx1, sy1, rgb in _spans_for_page(pageid):
            ox = _overlap(x0, x1, sx0, sx1)
            oy = _overlap(y0, y1, sy0, sy1)
            if ox <= 0 or oy <= 0:
                continue
            span_area = max(1e-6, (sx1 - sx0) * (sy1 - sy0))
            coverage = (ox * oy) / span_area  # fraction of the span inside the box
            # When two spans overlap (shadow case), break ties toward non-black,
            # since the visible foreground is the colored/white layer.
            is_black = all(v <= 0.01 for v in rgb)
            score = coverage + (0.0 if is_black else 0.15)
            if score > best_score:
                best_score = score
                best = rgb
        # Require a reasonably strong match before overriding pdfminer.
        if best is not None and best_score >= 0.5:
            return best
        return None

    return oracle


def _build_box_oracle(doc_zh):
    """Return f(pageid, x0, y0, x1, y1)->box_width|None.

    Snapshots the rectangle vector graphics (flowchart boxes / table cells) on
    each page and, for a query text position, returns the width of the smallest
    rectangle that encloses it. This is the *true* box width — used to fit
    translated figure labels precisely (the layout model only gives the whole
    figure, not each cell). Coords are converted to bottom-left to match
    pdfminer."""
    cache = {}
    # 每页的纵向网格线 x 坐标(列边界)。用于当某单元格未被任何"够大的矩形"包围时
    # (常见于只画边框线、不画填充矩形的表格行)，仍能由相邻列边界推断真实列宽。
    col_edges = {}
    try:
        for pageid in range(doc_zh.page_count):
            page = doc_zh[pageid]
            ph = page.rect.height
            rects = []
            edges = set()
            for dr in page.get_drawings():
                for it in dr.get("items", []):
                    if it[0] == "re":
                        r = it[1]
                        if r.width > 25 and r.height > 10:  # box-sized only
                            # top-left -> bottom-left
                            rects.append((r.x0, ph - r.y1, r.x1, ph - r.y0, r.width))
                        # 收集列边界：任何"非毛刺"矩形的左右边都算作潜在网格线，
                        # 包括 1px 细边框矩形(它们正是被上面尺寸过滤掉的表格线)。
                        if r.width > 2 or r.height > 2:
                            edges.add(round(r.x0, 1))
                            edges.add(round(r.x1, 1))
                    elif it[0] == "l":  # 直线边框(p1, p2)
                        try:
                            (px0, _py0), (px1, _py1) = it[1], it[2]
                            if abs(px1 - px0) < 1.0:  # 竖线
                                edges.add(round(px0, 1))
                        except Exception:
                            pass
            cache[pageid] = rects
            col_edges[pageid] = sorted(edges)
    except Exception:
        pass

    def _col_width(pageid, cx):
        """由列边界推断 cx 所在列的宽度(无包围矩形时的兜底)。"""
        es = col_edges.get(pageid)
        if not es:
            return None
        left = None
        right = None
        for e in es:
            if e <= cx + 0.5:
                left = e
            elif right is None and e > cx + 0.5:
                right = e
                break
        if left is not None and right is not None and right - left > 1:
            return right - left
        return None

    def oracle(pageid, x0, y0, x1, y1):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        best_w = None
        best_area = None
        for rx0, ry0, rx1, ry1, w in cache.get(pageid, []):
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                area = (rx1 - rx0) * (ry1 - ry0)
                if best_area is None or area < best_area:  # smallest enclosing
                    best_area = area
                    best_w = w
        if best_w is not None:
            return best_w
        # 兜底：无包围矩形(表格行只画边框线)时，按列边界估算列宽，
        # 避免短文本碎片(如换行残留的"degree")退化成极窄宽度而压垮整表字号。
        return _col_width(pageid, cx)

    return oracle


def _install_color_oracle(hl):
    """Wrap translate_patch so each converter gets PyMuPDF color + box oracles."""
    if getattr(hl, "_pdf2zh_patch_oracle_installed", False):
        return
    orig_translate_patch = hl.translate_patch
    PatchedConv = hl.TranslateConverter

    # Subclass that grabs doc_zh (captured per-call) at construction time.
    _ctx = {"doc_zh": None}

    class _OracleConverter(PatchedConv):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if _ctx["doc_zh"] is not None:
                self.color_oracle = _build_color_oracle(_ctx["doc_zh"])
                self.box_oracle = _build_box_oracle(_ctx["doc_zh"])

    def wrapped_translate_patch(*a, **k):
        _ctx["doc_zh"] = k.get("doc_zh")
        return orig_translate_patch(*a, **k)

    hl.TranslateConverter = _OracleConverter
    hl.translate_patch = wrapped_translate_patch
    hl._pdf2zh_patch_oracle_installed = True
