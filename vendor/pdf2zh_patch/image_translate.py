# -*- coding: utf-8 -*-
"""图像内文字翻译 (--translate-images)。

PDF 中的图表常把标题/坐标轴/图例文字"烧录"进位图(非文本层)，常规文本替换无法
触及。本模块对输出 PDF 做后处理：

  1. 检测：用 EasyOCR 在高 DPI 页面渲染上找出含目标源语言文字的方框(EasyOCR 框准、
     小字识别弱)。
  2. 识别+翻译：把每个方框裁剪、放大后交给视觉 LLM(GPT-4o)精确转写并翻译
     (视觉模型在整图上会臆造，但在紧致裁剪上很准)。
  3. 抹除+重绘：把原文像素抹成透明，按原色、自适应字号重绘译文。
  4. 回写：page.replace_image(xref) 原位替换像素，保留原 smask 透明度。

依赖(仅在启用本功能时才需要)：easyocr, torch, openai, Pillow, numpy, PyMuPDF。

研究记录见 docs/image-text-translation-research.md。
"""

import base64
import io
import os
import re

# CJK(中日韩)字符；拉丁字母。用于按源语言决定检测哪种文字。
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# 进程内缓存 EasyOCR reader 与逐框翻译，避免重复加载/重复调用。
_READER = None
_TRANS_CACHE = {}


def _has_cjk(s):
    return bool(_CJK_RE.search(s or ""))


def _get_reader(langs=("ch_sim", "en")):
    global _READER
    if _READER is None:
        import easyocr  # 重依赖：延迟导入
        _READER = easyocr.Reader(list(langs), gpu=False, verbose=False)
    return _READER


# 拉丁字体候选(目标为英文等)。
_LATIN_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
# 中日韩字体候选(目标为中/日/韩时必须用，否则汉字会渲染成方框)。
_CJK_FONTS = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


def _font(size, cjk=False):
    from PIL import ImageFont
    paths = (_CJK_FONTS + _LATIN_FONTS) if cjk else _LATIN_FONTS
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ocr_translate_crop(client, model, crop, lang_out):
    """对单个裁剪图(已放大)用视觉 LLM 转写源文字并翻译，返回译文(失败返回 None)。"""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        f"This is a tiny crop of a chart label. Transcribe ONLY the text exactly "
        f"as written (read the glyphs, do not guess from context), then ' | ' then "
        f"a SHORT natural {lang_out} translation. "
        f"Keep numbers, percent signs (%), units, and symbols EXACTLY as written — "
        f"e.g. '88%' stays '88%', never spell it out as '88 percent'. "
        f"If the text is purely a number/percentage/symbol, the translation is identical to it. "
        f"If there is no readable text, reply 'NONE'. Output one line, nothing else."
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ]}],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip()
    except Exception:
        return None
    if not out or out.upper().startswith("NONE"):
        return None
    eng = out.split("|")[-1].strip() if "|" in out else out
    # 去掉可能的引号/前后噪声
    eng = eng.strip().strip('"').strip()
    return eng or None


def _batch_translate(client, model, texts, lang_out):
    """一次性翻译一组短文本(用于旋转表格的整块)。逐项命中缓存，未命中的合并为一次
    文本调用(非视觉)，比逐格视觉调用快得多。返回与 texts 等长的译文列表。"""
    import json
    results = [None] * len(texts)
    todo = []  # (index, text)
    for i, t in enumerate(texts):
        c = _TRANS_CACHE.get(t.strip())
        if c is not None:
            results[i] = c
        elif t.strip():
            todo.append((i, t))
    if todo:
        numbered = "\n".join(f"{k}\t{t}" for k, (_, t) in enumerate(todo))
        prompt = (
            f"Translate each numbered line below to {lang_out}. These are cells of a "
            f"table. Keep numbers, %, dates, codes, and symbols EXACTLY as written. "
            f"Return JSON {{\"t\":[\"...\",...]}} with translations in the SAME order, "
            f"same count. Output JSON only."
            f"\n\n{numbered}"
        )
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, response_format={"type": "json_object"},
            )
            arr = json.loads(r.choices[0].message.content).get("t", [])
        except Exception:
            arr = []
        for k, (i, t) in enumerate(todo):
            eng = arr[k].strip() if k < len(arr) and isinstance(arr[k], str) else t
            results[i] = eng
            _TRANS_CACHE[t.strip()] = eng
    return results


def _text_color(arr, x0, y0, x1, y1):
    """方框内文字颜色：背景=不透明像素的中位色(占多数)；文字=离背景最远的那批像素的
    代表色。这样无论文字是黑/蓝/红/绿都能正确取到，且不会误取浅色背景或抗锯齿边缘。"""
    import numpy as np
    sub = arr[y0:y1, x0:x1].reshape(-1, 4).astype(int)
    op = sub[sub[:, 3] > 128][:, :3]
    if len(op) == 0:
        return (60, 60, 60, 255)
    bg = np.median(op, axis=0)                       # 背景色(多数像素)
    dist = np.abs(op - bg).sum(axis=1)               # 每个像素与背景的距离
    far_thr = max(60, dist.max() * 0.6)              # 取"明显非背景"的像素(文字+其浓边)
    fg_pixels = op[dist >= far_thr]
    if len(fg_pixels) == 0:
        fg_pixels = op[dist.argmax()][None, :]
    # 文字核心色：这些前景像素里"最浓"(离背景最远)的中位色，避免抗锯齿浅边拉淡。
    d2 = np.abs(fg_pixels - bg).sum(axis=1)
    core = fg_pixels[d2 >= np.percentile(d2, 70)]
    rep = np.median(core, axis=0).astype(int) if len(core) else fg_pixels[d2.argmax()].astype(int)
    return (int(rep[0]), int(rep[1]), int(rep[2]), 255)


def _translate_image(doc, page, im, client, model, lang_out, reader,
                     src_is_cjk=True, tgt_is_cjk=False, min_conf=0.0):
    """翻译单张嵌入图像中的文字。返回替换的文字块数。

    src_is_cjk: 源语言是否中日韩(决定检测哪种文字)；
    tgt_is_cjk: 目标语言是否中日韩(决定重绘用 CJK 字体)。"""
    import fitz
    from PIL import Image, ImageDraw
    import numpy as np

    def _is_src(text):
        # 源文字识别：CJK 源 -> 含 CJK；拉丁源(英文等) -> 含 >=2 个拉丁字母且不含 CJK。
        if src_is_cjk:
            return _has_cjk(text)
        return (not _has_cjk(text)) and len(_LATIN_LETTER_RE.findall(text or "")) >= 2

    xref = im[0]
    smask = im[1]
    rects = page.get_image_rects(xref)
    if not rects:
        return 0

    # 取"应用 smask 后"的真实外观 (RGBA)，否则背景为不透明黑会干扰取色/抹除。
    base = fitz.Pixmap(doc, xref)
    if base.n - base.alpha >= 4:  # CMYK 等 -> 转 RGB
        base = fitz.Pixmap(fitz.csRGB, base)
    if smask:
        try:
            base = fitz.Pixmap(base, fitz.Pixmap(doc, smask))
        except Exception:
            pass
    mode = "RGBA" if base.alpha else "RGB"
    img = Image.frombytes(mode, [base.width, base.height], base.samples).convert("RGBA")
    OW, OH = img.size
    if OW < 8 or OH < 8:
        return 0

    # 在"高 DPI 页面渲染"上做检测(比低分嵌入图清晰得多)。把图像 rect 渲染出来。
    rect = rects[0]
    zoom = 4.0  # 渲染倍率；4x 足以让 EasyOCR 框准小字
    mat = fitz.Matrix(zoom, zoom)
    hp = page.get_pixmap(matrix=mat, clip=rect)
    hidpi = Image.frombytes("RGB", [hp.width, hp.height], hp.samples)
    HW, HH = hidpi.size
    sx, sy = OW / HW, OH / HH

    # EasyOCR：开启 rotation_info 以便检测竖排/旋转文字框(其识别仍弱，靠 GPT-4o 兜底)。
    try:
        results = reader.readtext(np.array(hidpi), detail=1, rotation_info=[90, 270])
    except Exception:
        results = reader.readtext(np.array(hidpi), detail=1)

    # 归一化为 (HX0,HY0,HX1,HY1, raw, conf)
    dets = []
    for box, raw, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        dets.append([min(xs), min(ys), max(xs), max(ys), raw, conf])

    def _overlap(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        return ix > -6 and iy > 0  # 同一行且横向相邻/相交(留 6px 容差)

    # 标记需要处理的检测：含源语言文字 -> 翻译；与之同行相邻、属于"另一种文字"的框
    #  -> 并入源语言框的包围盒(其内容会被相邻译文覆盖，如标题里夹带的 "AI"；并入后统一
    # 抹除+重绘，避免单独抹除把已绘译文打出空洞)；竖高窄的空/低置信框 -> 旋转重识别。
    src_idx = [i for i, d in enumerate(dets) if _is_src(d[4]) and d[5] >= min_conf]
    merged = set()
    for j, d in enumerate(dets):
        if j in src_idx or _is_src(d[4]):
            continue
        for i in src_idx:
            if _overlap(d, dets[i]):
                dets[i][0] = min(dets[i][0], d[0])
                dets[i][1] = min(dets[i][1], d[1])
                dets[i][2] = max(dets[i][2], d[2])
                dets[i][3] = max(dets[i][3], d[3])
                merged.add(j)
                break

    arr = np.array(img)
    draw = ImageDraw.Draw(img)
    n = 0

    def _erase(x0, y0, x1, y1):
        for yy in range(max(0, y0 - 1), min(OH, y1 + 1)):
            for xx in range(max(0, x0 - 1), min(OW, x1 + 1)):
                img.putpixel((xx, yy), (0, 0, 0, 0))

    def _fit_size(eng, bw, bh):
        """求能把 eng 放进 (bw 的 1.5 倍, bh 的 1.25 倍) 的最大字号。"""
        max_w = bw * 1.5
        size = bh + 2
        while size > 5:
            f = _font(size, cjk=tgt_is_cjk); tb = draw.textbbox((0, 0), eng, font=f)
            if tb[2] - tb[0] <= max_w and tb[3] - tb[1] <= bh * 1.25:
                return size
            size -= 1
        return max(5, size)

    def _draw_fit(eng, x0, y0, x1, y1, fg, vertical=False, fixed_size=None):
        bw, bh = x1 - x0, y1 - y0
        if vertical:
            bw, bh = bh, bw  # 文字按水平排版后再旋转，故宽高互换
        size = fixed_size if fixed_size else _fit_size(eng, bw, bh)
        f = _font(size, cjk=tgt_is_cjk); tb = draw.textbbox((0, 0), eng, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if vertical:
            # 在透明小图上水平绘制，再旋转 90° 贴回(竖排标签，如 Y 轴标题)
            pad = 4
            tile = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((pad - tb[0], pad - tb[1]), eng, font=f, fill=fg)
            tile = tile.rotate(90, expand=True)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            img.alpha_composite(tile, (max(0, cx - tile.width // 2), max(0, cy - tile.height // 2)))
        else:
            cx = (x0 + x1) // 2
            tx = max(2, min(cx - tw // 2, OW - tw - 2))
            draw.text((tx, y0 + ((y1 - y0) - th) // 2 - tb[1]), eng, font=f, fill=fg)

    # 第一遍：识别+翻译，收集待绘制项 (不立即绘制，以便先统一同类标签字号)。
    items = []  # (x0,y0,x1,y1, eng, is_vert, fg, fit_size)
    for i, d in enumerate(dets):
        HX0, HY0, HX1, HY1, raw, conf = d
        x0, y0 = int(HX0 * sx), int(HY0 * sy)
        x1, y1 = int(HX1 * sx), int(HY1 * sy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        w, h = HX1 - HX0, HY1 - HY0
        is_vert = h > w * 1.3  # 竖高窄 -> 疑似竖排
        if i in merged:
            continue
        if i not in src_idx and not (is_vert and not raw.strip()):
            continue
        crop = hidpi.crop((max(0, HX0 - 10), max(0, HY0 - 10),
                           min(HW, HX1 + 10), min(HH, HY1 + 10)))
        if is_vert:
            crop = crop.rotate(-90, expand=True)
        key = raw.strip() or f"vert@{x0},{y0}"
        eng = _TRANS_CACHE.get(key)
        if eng is None:
            eng = _ocr_translate_crop(client, model, crop, lang_out)
            if eng:
                _TRANS_CACHE[key] = eng
        if not eng:
            continue
        fg = _text_color(arr, x0, y0, x1, y1)
        bw, bh = (x1 - x0, y1 - y0) if not is_vert else (y1 - y0, x1 - x0)
        items.append([x0, y0, x1, y1, eng, is_vert, fg, _fit_size(eng, bw, bh)])

    if not items:
        return 0

    # 统一同类标签字号：把"同一类"标签(横排、原框高度相近、左边缘相近——典型如一组
    # 坐标轴类目标签)归为一组，取组内最小字号，避免长短标签字号忽大忽小、长标签压到图。
    horiz = [it for it in items if not it[5]]
    used = [False] * len(horiz)
    for a in range(len(horiz)):
        if used[a]:
            continue
        grp = [a]
        ha = horiz[a][3] - horiz[a][1]
        for b in range(a + 1, len(horiz)):
            if used[b]:
                continue
            hb = horiz[b][3] - horiz[b][1]
            same_h = abs(ha - hb) <= max(4, 0.35 * ha)
            same_x = abs(horiz[a][0] - horiz[b][0]) <= max(8, 0.05 * OW)
            if same_h and same_x:
                grp.append(b)
        if len(grp) >= 2:
            gsize = min(horiz[g][7] for g in grp)
            for g in grp:
                horiz[g][7] = gsize
                used[g] = True
        else:
            used[a] = True

    # 第二遍：抹除+绘制(用统一后的字号)。
    for x0, y0, x1, y1, eng, is_vert, fg, fsize in items:
        _erase(x0, y0, x1, y1)
        _draw_fit(eng, x0, y0, x1, y1, fg, vertical=is_vert, fixed_size=fsize)
        n += 1

    if n == 0:
        return 0
    # 原位替换像素，PyMuPDF 会按 PNG 透明度重建 mask，保留透明背景。
    out = io.BytesIO()
    img.save(out, format="PNG")
    try:
        page.replace_image(xref, stream=out.getvalue())
    except Exception:
        return 0
    return n


def translate_images_in_pdf(pdf_path, lang_in, lang_out, service, model,
                            api_key=None, base_url=None,
                            orig_pdf_path=None, rotated_regions=None):
    """对 PDF 内所有嵌入图像翻译其源语言文字，原地保存。返回 (图像数, 替换块数)。

    支持 中日韩 <-> 英文 等方向(源与目标须为不同文字体系) + OpenAI 兼容的视觉模型。
    其它情况安全跳过(返回 0,0)。

    orig_pdf_path/rotated_regions: 若提供，则同时翻译"旋转文字区域"(如横放表格)——
    这些文字已在转换阶段被跳过(否则会被直立重排成乱码)，此处从原始页面光栅化后
    OCR+翻译并叠加到输出页面。rotated_regions: {pageid: [(x0,y0,x1,y1)...]}。"""
    if service.split(":")[0] not in ("openai", "azure-openai"):
        print("⏭️  图像翻译目前仅支持 openai 视觉模型，已跳过 (--service openai)")
        return (0, 0)
    src_is_cjk = _lang_is_cjk(lang_in)
    tgt_is_cjk = _lang_is_cjk(lang_out)
    # 需要源/目标分属不同文字体系(CJK <-> 非 CJK)，否则无从判断哪些是源文字。
    if src_is_cjk == tgt_is_cjk:
        print("⏭️  图像翻译目前支持 中日韩↔英文 方向，当前语言组合已跳过")
        return (0, 0)

    try:
        import fitz
        from openai import OpenAI
    except Exception as e:
        print(f"⏭️  图像翻译缺少依赖，已跳过: {e}")
        return (0, 0)
    try:
        reader = _get_reader()
    except Exception as e:
        print(f"⏭️  无法加载 EasyOCR (pip install easyocr)，已跳过: {e}")
        return (0, 0)

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
    )

    doc = fitz.open(pdf_path)
    orig_doc = None
    if orig_pdf_path and rotated_regions:
        try:
            orig_doc = fitz.open(orig_pdf_path)
        except Exception:
            orig_doc = None
    n_imgs = 0
    n_blocks = 0
    for pid in range(doc.page_count):
        page = doc[pid]
        for im in page.get_images(full=True):
            try:
                replaced = _translate_image(doc, page, im, client, model, lang_out, reader,
                                            src_is_cjk=src_is_cjk, tgt_is_cjk=tgt_is_cjk)
            except Exception:
                replaced = 0
            if replaced:
                n_imgs += 1
                n_blocks += replaced
        # 旋转文字区域(横放表格等)：从原始页面光栅化后翻译并叠加。
        if orig_doc is not None and rotated_regions.get(pid):
            try:
                rb = _translate_rotated_on_page(
                    orig_doc[pid], page, rotated_regions[pid], client, model,
                    lang_out, reader, src_is_cjk=src_is_cjk, tgt_is_cjk=tgt_is_cjk)
            except Exception:
                rb = 0
            if rb:
                n_imgs += 1
                n_blocks += rb
    if orig_doc is not None:
        orig_doc.close()
    if n_blocks:
        tmp = pdf_path + ".imgtmp.pdf"
        doc.save(tmp, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp, pdf_path)
    else:
        doc.close()
    return (n_imgs, n_blocks)


def _cluster_boxes(boxes, gap=6.0):
    """把许多字符级包围盒合并成若干"块"(简单的并查/外扩合并)。boxes: (x0,y0,x1,y1)。
    返回合并后的块包围盒列表。用于把旋转表格的零散字符聚成可处理的区域。"""
    rects = [list(b) for b in boxes]
    changed = True
    while changed and len(rects) > 1:
        changed = False
        out = []
        for r in rects:
            merged_into = None
            for o in out:
                # 外扩 gap 后相交即合并
                if (r[0] <= o[2] + gap and r[2] >= o[0] - gap and
                        r[1] <= o[3] + gap and r[3] >= o[1] - gap):
                    o[0] = min(o[0], r[0]); o[1] = min(o[1], r[1])
                    o[2] = max(o[2], r[2]); o[3] = max(o[3], r[3])
                    merged_into = o; changed = True; break
            if merged_into is None:
                out.append(r[:])
        rects = out
    return rects


def _translate_rotated_on_page(orig_page, out_page, char_boxes, client, model,
                               lang_out, reader, src_is_cjk, tgt_is_cjk):
    """翻译某页"旋转文字"区域：从原始页面光栅化每个块，按旋转后水平方向 OCR+翻译，
    再把译文(旋转回原方向)叠加到输出页面。返回处理的文本块数。"""
    import fitz
    from PIL import Image, ImageDraw
    import numpy as np

    PH = orig_page.rect.height
    # 字符盒(pdfminer 坐标, y 自下而上) -> PyMuPDF 坐标(y 自上而下)
    boxes_tl = [(x0, PH - y1, x1, PH - y0) for (x0, y0, x1, y1) in char_boxes]
    blocks = _cluster_boxes(boxes_tl, gap=8.0)
    # 仅保留足够大的块(过滤零散噪声)
    blocks = [b for b in blocks if (b[2] - b[0]) > 30 and (b[3] - b[1]) > 30]
    if not blocks:
        return 0

    n = 0
    for bx0, by0, bx1, by1 in blocks:
        rect = fitz.Rect(bx0, by0, bx1, by1)
        zoom = 4.0
        hp = orig_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
        crop = Image.frombytes("RGB", [hp.width, hp.height], hp.samples)
        # 该块文字是顺时针旋转 90°(dir=(0,-1))写的，旋转 +90° 使其水平以便 OCR。
        upright = crop.rotate(-90, expand=True)
        arr = np.array(upright)
        try:
            res = reader.readtext(arr, detail=1)
        except Exception:
            res = []
        UW, UH = upright.size
        # 收集该块所有源文字框，一次性批量翻译(每块 1 次 LLM 调用，避免逐格调用过慢)。
        raw_items = []
        for box, raw, conf in res:
            if not _src_text(raw, src_is_cjk):
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            raw_items.append((min(xs), min(ys), max(xs), max(ys), raw.strip()))
        if not raw_items:
            continue
        texts = [it[4] for it in raw_items]
        trans = _batch_translate(client, model, texts, lang_out)
        items = []
        for (ux0, uy0, ux1, uy1, raw), eng in zip(raw_items, trans):
            if eng:
                items.append((ux0, uy0, ux1, uy1, eng))
        if not items:
            continue

        # 把译文画到一张 upright 画布上(白底→后续旋转回去叠加)，逐块用透明底以便旋转贴回。
        tile = Image.new("RGBA", (UW, UH), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        # 先在 upright 上把原文区域抹白(覆盖原英文)，再写译文。
        cover = Image.new("RGBA", (UW, UH), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cover)
        for ux0, uy0, ux1, uy1, eng in items:
            cd.rectangle([ux0 - 2, uy0 - 2, ux1 + 2, uy1 + 2], fill=(255, 255, 255, 255))
            bh = uy1 - uy0
            size = max(6, int(bh) + 1)
            while size > 5:
                f = _font(size, cjk=tgt_is_cjk); tb = td.textbbox((0, 0), eng, font=f)
                if tb[2] - tb[0] <= (ux1 - ux0) * 1.6 and tb[3] - tb[1] <= bh * 1.3:
                    break
                size -= 1
            f = _font(size, cjk=tgt_is_cjk); tb = td.textbbox((0, 0), eng, font=f)
            td.text((ux0 - tb[0], uy0 + (bh - (tb[3] - tb[1])) // 2 - tb[1]), eng,
                    font=f, fill=(20, 20, 20, 255))
        # 合成：白底覆盖 + 译文，再旋转回 -90°(即顺时针 90°)对齐原方向
        merged = Image.alpha_composite(cover, tile)
        back = merged.rotate(90, expand=True)
        # 贴回输出页面对应矩形(insert_image 接受 PNG 流，铺满该 rect)
        buf = io.BytesIO(); back.save(buf, format="PNG")
        try:
            out_page.insert_image(rect, stream=buf.getvalue(), overlay=True)
            n += len(items)
        except Exception:
            pass
    return n


def _src_text(text, src_is_cjk):
    if src_is_cjk:
        return _has_cjk(text)
    return (not _has_cjk(text)) and len(_LATIN_LETTER_RE.findall(text or "")) >= 2


def _lang_is_cjk(lang):
    return str(lang).lower() in {
        "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "zh-hans-cn", "chinese",
        "ja", "japanese", "ko", "korean",
    }
