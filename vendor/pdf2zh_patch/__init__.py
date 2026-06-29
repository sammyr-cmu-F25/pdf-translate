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
import re
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
    _install_translate_guard()

    if translate_figures:
        _install_figure_unlock()

    return True


# PATCH#7 译文净化：LLM 类服务(OpenAI 等)在收到"空白/纯数字/纯符号"等无需翻译的
# 片段时，常返回对话式拒绝语("Please provide the source text…"/"I'm sorry…")，这些
# 串会被当作译文写入 PDF，污染表格空单元格(如财务表)并因过长而溢出方框。此守卫：
#   1. 对空白/纯数字/纯标点/纯公式占位的输入直接返回原文，不送 LLM；
#   2. 若 LLM 仍返回拒绝/澄清式回答，回退为原文。
_REFUSAL_PAT = re.compile(
    r"(provide the source|no source text|i'?m sorry|as an ai|cannot translate|"
    r"unable to translate|please (provide|share|give)|there is no (source|text)|"
    r"could you (please )?(provide|clarify)|i (can'?t|cannot) (assist|help|translate))",
    re.IGNORECASE,
)
# 只含数字、空白、标点、公式占位 {v..} 的文本无需翻译。
_NO_TRANSLATE_PAT = re.compile(r"^[\s\d\W]*$|^(?:\s*\{\s*v[\d\s]+\}\s*)+$", re.IGNORECASE)

# LLM 误回显的提示词前缀(出现在译文开头或独立成段)。
_PROMPT_ECHO_PAT = re.compile(
    r"^\s*(translated\s+text|translation|译文|翻译(结果)?|target\s+text)\s*[:：]\s*",
    re.IGNORECASE,
)
# LLM 对"截断片段"添加的元注释(说明原文不完整/以省略号结尾等)。这类句子不是译文，
# 应整句删除。匹配任一以句界为边界、且谈及 source text / incomplete / ellipsis /
# truncated / the translation will reflect / 原文…不完整/省略号 的句子(可带 Note:/注:
# 前缀)。逐句剥离。
_META_COMMENT_PAT = re.compile(
    r"(?:^|(?<=[.。!?！？]))\s*"
    r"(?:(?:note|注|说明)\s*[:：]?\s*)?"
    r"[^.。!?！？]*?"
    r"(the source text|source text is|is incomplete|ends with (an )?ellipsis|"
    r"truncat\w*|translation will reflect|省略号|原文[^.。!?！？]*不完整)"
    r"[^.。!?！？]*[.。!?！？]?",
    re.IGNORECASE,
)

# CJK 字符(中日韩表意文字)。译文若目标语言非 CJK 却仍含这些字符，说明 LLM 漏译。
_CJK_PAT = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_CJK_TARGETS = {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "ja", "ko",
                "chinese", "japanese", "korean", "zh-hans-cn"}

# 收集"漏译"片段(目标语言非 CJK 但译文仍含 CJK)，运行结束后统一提示。
LEFTOVER_CJK = []


def _cjk_ratio(s):
    cj = len(_CJK_PAT.findall(s))
    tot = len(re.sub(r"\s", "", s)) or 1
    return cj / tot


def _install_translate_guard():
    """包裹 BaseTranslator.translate，过滤无需翻译的片段、净化 LLM 拒绝语，
    并对"目标语言非 CJK 却仍含中日韩字符"的漏译片段重试一次、最终仍漏则记录待提示。"""
    try:
        import pdf2zh.translator as tr
    except Exception:
        return
    Base = getattr(tr, "BaseTranslator", None)
    if Base is None or getattr(Base, "_pdf2zh_patch_guard", False):
        return
    orig_translate = Base.translate

    def guarded_translate(self, text, ignore_cache=False):
        # 空白 / 纯数字符号 / 纯公式占位：原样返回，不调用翻译服务。
        if text is None:
            return text
        if not text.strip() or _NO_TRANSLATE_PAT.match(text):
            return text
        out = orig_translate(self, text, ignore_cache)
        # 清理 LLM 对截断片段添加的元注释(如 "Note: The source text is incomplete…")，
        # 以及误回显的提示词前缀(如 "Translation:")。仅当原文本身不含这些时才剥离，
        # 避免误删正文。
        if isinstance(out, str):
            if not _META_COMMENT_PAT.search(text or ""):
                cleaned = _META_COMMENT_PAT.sub("", out).strip()
                if cleaned != out:
                    out = cleaned
            if not _PROMPT_ECHO_PAT.match(text or ""):
                m = _PROMPT_ECHO_PAT.match(out)
                if m:
                    rest = out[m.end():].strip()
                    # 前缀后有正文 -> 保留正文；整段仅为前缀噪声 -> 置空(不渲染多余文字)。
                    out = rest if rest else ""
        # LLM 返回了拒绝/澄清式回答(且原文并非本就如此)：回退原文，避免污染版面。
        if isinstance(out, str) and _REFUSAL_PAT.search(out) and not _REFUSAL_PAT.search(text):
            return text

        # 漏译检测与重试：目标语言非 CJK，但译文仍残留任何中日韩字符即视为漏译
        # (英文等输出本不应含 CJK)。哪怕只漏一个词(如句首"断天然配体"残留)也要处理。
        target_cjk = str(getattr(self, "lang_out", "")).lower() in _CJK_TARGETS
        if isinstance(out, str) and not target_cjk and _CJK_PAT.search(out):
            for _ in range(2):  # 重试至多两次(直接调 do_translate，绕过缓存重新请求)
                try:
                    retry = self.do_translate(text)
                except Exception:
                    retry = None
                if isinstance(retry, str) and retry.strip() and not _CJK_PAT.search(retry):
                    self.cache.set(text, retry)  # 干净结果回写缓存
                    return retry
                if isinstance(retry, str) and retry.strip() and _cjk_ratio(retry) < _cjk_ratio(out):
                    out = retry  # 保留更干净的一次
            # 重试后仍残留 CJK：记录(原文, 译文)，运行结束统一提示人工复核。
            LEFTOVER_CJK.append((text[:60], out[:60]))
        return out

    Base.translate = guarded_translate
    Base._pdf2zh_patch_guard = True


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
            # 边界聚类：表格常有反锯齿双边、装饰细线，产生大量相距 <几 pt 的近重复
            # 边。直接相邻取差会把"列宽"误判成毛刺间隙(如 3pt)。先把相距 <=_EDGE_TOL
            # 的边合并成一条(取其均值)，再作为真正的列边界。
            _EDGE_TOL = 4.0
            es = sorted(edges)
            clustered = []
            for e in es:
                if clustered and e - clustered[-1][-1] <= _EDGE_TOL:
                    clustered[-1].append(e)
                else:
                    clustered.append([e])
            col_edges[pageid] = [sum(g) / len(g) for g in clustered]
    except Exception:
        pass

    # 列宽下限：窄于此值的"列"几乎一定是毛刺间隙而非真实单元格，
    # 此时忽略该候选，避免给出过窄宽度把文字压到极小。
    _MIN_COL_W = 15.0

    def _col_width(pageid, qx0, qx1):
        """由列边界推断查询单元格的"可用横向宽度"——从单元格文本左边 qx0 到所在列
        右边界的距离。译文按单元格左边锚定渲染，故用 (列右界 - 单元格左边) 才能保证
        换行后不越过列右界压到右侧相邻列(严禁重叠)；若返回整列宽(列右界-列左界)，
        当文本在源文档中居中(qx0 > 列左界)时仍会向右溢出。

        列左界 = 紧贴 qx0 及更左的最近网格线；列右界 = 位于单元格右边之外的最近网格
        线。要求结果 >= _MIN_COL_W，否则视为毛刺，返回 None。"""
        es = col_edges.get(pageid)
        if not es:
            return None
        left = None
        for e in es:
            if e <= qx0 + 1.0:
                left = e
            else:
                break
        right = None
        for e in es:
            # 列右界须在单元格右边(含少量容差)之外
            if e >= qx1 - 1.0:
                right = e
                if left is None or right - max(left, qx0) >= _MIN_COL_W:
                    break
        if left is not None and right is not None:
            usable = right - max(left, qx0)
            if usable >= _MIN_COL_W:
                return usable
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
        col_w = _col_width(pageid, x0, x1)
        if best_w is not None:
            # 包围矩形常常是整张表的外框(内部单元格未单独描边)，其宽度远大于真实
            # 列宽，会导致译文不缩小而横向压到相邻单元格(严禁重叠)。若由网格线推断出
            # 的列宽明显更窄，则以列宽为准。
            if col_w is not None and col_w < best_w - 1:
                return col_w
            return best_w
        # 无包围矩形(表格行只画边框线)时，按列边界估算列宽，避免短文本碎片
        # (如换行残留的"degree")退化成极窄宽度而压垮整表字号。
        return col_w

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
