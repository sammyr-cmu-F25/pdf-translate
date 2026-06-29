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

# CJK(中日韩)字符。目前检测以 CJK 源语言为主(中→英等)。
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


def _font(size):
    from PIL import ImageFont
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
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
        f"a SHORT natural {lang_out} translation. If there is no readable text, "
        f"reply 'NONE'. Output one line, nothing else."
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


def _translate_image(doc, page, im, client, model, lang_out, reader, min_conf=0.0):
    """翻译单张嵌入图像中的文字。返回替换的文字块数。"""
    import fitz
    from PIL import Image, ImageDraw
    import numpy as np

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

    results = reader.readtext(np.array(hidpi), detail=1)

    arr = np.array(img)
    draw = ImageDraw.Draw(img)
    n = 0
    for box, raw, conf in results:
        if conf < min_conf or not _has_cjk(raw):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        HX0, HY0, HX1, HY1 = min(xs), min(ys), max(xs), max(ys)
        # 用高 DPI 清晰裁剪做识别+翻译
        crop = hidpi.crop((max(0, HX0 - 10), max(0, HY0 - 10),
                           min(HW, HX1 + 10), min(HH, HY1 + 10)))
        key = raw.strip()
        eng = _TRANS_CACHE.get(key)
        if eng is None:
            eng = _ocr_translate_crop(client, model, crop, lang_out)
            if eng:
                _TRANS_CACHE[key] = eng
        if not eng:
            continue
        # 映射回原图像素坐标
        x0, y0 = int(HX0 * sx), int(HY0 * sy)
        x1, y1 = int(HX1 * sx), int(HY1 * sy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        fg = _text_color(arr, x0, y0, x1, y1)
        bw, bh = x1 - x0, y1 - y0
        # 抹除原文像素(透明)
        for yy in range(max(0, y0 - 1), min(OH, y1 + 1)):
            for xx in range(max(0, x0 - 1), min(OW, x1 + 1)):
                img.putpixel((xx, yy), (0, 0, 0, 0))
        # 自适应字号：宽度不超过 1.5×原框(居中标签留些余量)，高度不超过 1.25×
        max_w = bw * 1.5
        size = bh + 2
        while size > 5:
            f = _font(size)
            tb = draw.textbbox((0, 0), eng, font=f)
            if tb[2] - tb[0] <= max_w and tb[3] - tb[1] <= bh * 1.25:
                break
            size -= 1
        f = _font(size)
        tb = draw.textbbox((0, 0), eng, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        cx = (x0 + x1) // 2
        tx = max(2, min(cx - tw // 2, OW - tw - 2))
        draw.text((tx, y0 + (bh - th) // 2 - tb[1]), eng, font=f, fill=fg)
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
                            api_key=None, base_url=None):
    """对 PDF 内所有嵌入图像翻译其源语言文字，原地保存。返回 (图像数, 替换块数)。

    目前仅支持源语言为中日韩(CJK)的图像文字 + OpenAI 兼容的视觉模型。
    其它情况安全跳过(返回 0,0)。"""
    if service.split(":")[0] not in ("openai", "azure-openai"):
        print("⏭️  图像翻译目前仅支持 openai 视觉模型，已跳过 (--service openai)")
        return (0, 0)
    if not _has_cjk_source(lang_in):
        print("⏭️  图像翻译目前仅支持中日韩源语言，已跳过")
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
    n_imgs = 0
    n_blocks = 0
    for pid in range(doc.page_count):
        page = doc[pid]
        for im in page.get_images(full=True):
            try:
                replaced = _translate_image(doc, page, im, client, model, lang_out, reader)
            except Exception:
                replaced = 0
            if replaced:
                n_imgs += 1
                n_blocks += replaced
    if n_blocks:
        tmp = pdf_path + ".imgtmp.pdf"
        doc.save(tmp, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp, pdf_path)
    else:
        doc.close()
    return (n_imgs, n_blocks)


def _has_cjk_source(lang_in):
    return str(lang_in).lower() in {
        "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "chinese",
        "ja", "japanese", "ko", "korean",
    }
