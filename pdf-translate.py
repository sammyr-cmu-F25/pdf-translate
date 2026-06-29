#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 翻译工具 — 基于 pdf2zh (PDFMathTranslate)，并加载本仓库的增强补丁。

保留原始排版、图形、表格，仅替换文本为目标语言。相比上游 pdf2zh 额外支持：
    PATCH#1  译文颜色与原文一致（修复白色标题翻译后变黑看不清的问题）
    PATCH#3  译文过长时自动缩小字号以塞进标题 / 表格单元格（避免溢出）

用法:
    python pdf-translate.py input.pdf                       # 默认英文→中文
    python pdf-translate.py input.pdf -li ja -lo zh         # 日文→中文
    python pdf-translate.py input.pdf -li zh -lo en         # 中文→英文
    python pdf-translate.py input.pdf --service openai      # 指定翻译服务
    python pdf-translate.py input.pdf -o /path/to/outdir    # 指定输出目录

依赖:
    Python 3.10–3.12（注意：3.13 上 pdf2zh 不兼容）
    pip install pdf2zh

输出:
    input-mono.pdf  — 纯目标语言版
    input-dual.pdf  — 双语对照版
"""

import argparse
import os
import sys

# 让 vendor/ 下的补丁可被导入
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "vendor"))


def main():
    os.environ.setdefault("PYTHONUTF8", "1")
    # macOS 上 ONNXRuntime 与 PyTorch 各自携带一份 libomp，重复加载会在解释器退出时
    # 触发段错误(translate 已完成但 PDF 尚未写盘 → 输出丢失)。允许重复加载并限制
    # OpenMP 线程数可避免该崩溃。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    parser = argparse.ArgumentParser(description="Translate a PDF while preserving layout (patched pdf2zh).")
    parser.add_argument("input", help="Path to the input PDF")
    parser.add_argument("-li", "--lang-in", default="en", help="Source language code (default: en)")
    parser.add_argument("-lo", "--lang-out", default="zh", help="Target language code (default: zh)")
    parser.add_argument("-s", "--service", default="google", help="Translation service (default: google)")
    parser.add_argument("-m", "--model", default="",
                        help="Model name for LLM services (default: gpt-4o for openai). "
                             "gpt-4o leaves far fewer untranslated fragments than gpt-4o-mini.")
    parser.add_argument("-o", "--output", default="", help="Output directory (default: alongside input)")
    parser.add_argument("-t", "--thread", type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument("--no-patch", action="store_true", help="Disable enhancement patches (use stock pdf2zh)")
    parser.add_argument("--protect-figures", action="store_true",
                        help="Do NOT translate text inside figures/tables (restore stock pdf2zh behavior). "
                             "By default this tool translates that text too (math formulas always preserved).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore the translation cache and re-translate everything from scratch.")
    parser.add_argument("--translate-images", action="store_true",
                        help="Also translate text baked into chart/figure bitmaps (OCR + redraw). "
                             "Requires --service openai; works CJK<->English (source and target "
                             "must differ in script); needs easyocr. "
                             "Adds an OCR model load + a vision call per detected label.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return 1

    # LLM 服务的模型选择：默认用 gpt-4o(比 gpt-4o-mini 漏译/残留中日韩字符少得多)。
    # 通过环境变量覆盖 pdf2zh 配置(set_envs 会优先读取 os.environ)。
    _service_base = args.service.split(":")[0]
    if _service_base in ("openai", "azure-openai"):
        chosen_model = args.model or "gpt-4o"
        os.environ["OPENAI_MODEL"] = chosen_model
        print(f"🧠 模型: {chosen_model}")
    elif args.model:
        # 其它 LLM 服务也允许显式指定模型
        os.environ.setdefault("OPENAI_MODEL", args.model)

    # 应用补丁（必须在 high_level 构建 converter 之前）
    if not args.no_patch:
        import pdf2zh_patch
        translate_figures = not args.protect_figures
        pdf2zh_patch.patch(translate_figures=translate_figures)
        extra = "" if not translate_figures else " + 翻译图表内文字(#2)"
        print(f"🩹 已加载增强补丁: 颜色匹配(#1) + 自适应字号(#3){extra}")

    import pdf2zh.high_level as hl
    from pdf2zh.doclayout import OnnxModel

    output_dir = args.output or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔄 翻译中: {os.path.basename(args.input)}")
    print(f"   {args.lang_in} → {args.lang_out} (服务: {args.service})")

    # 为 LLM 类服务(openai 等)提供更明确的上下文提示，减少脱离上下文的歧义翻译，
    # 例如把数量+单位 "12月+" 误译为月份 "December+"（应为 "12 months+"）。
    from string import Template
    llm_prompt = None
    if args.service.split(":")[0] in ("openai", "azure-openai", "deepseek", "grok",
                                      "groq", "gemini", "zhipu", "ollama", "silicon",
                                      "qwen-mt", "xinference", "openailiked"):
        llm_prompt = Template(
            "You are a professional translation engine for technical and business "
            "documents. Translate the source text from $lang_in to $lang_out.\n"
            "Rules:\n"
            "- Keep the formula notation {v*} unchanged.\n"
            "- Translate a number followed by a unit literally as a quantity, NOT as "
            "a date. For example Chinese \"12月+\" means \"12 months+\" (a duration), "
            "not the month \"December\"; \"3月\" in a duration context means "
            "\"3 months\".\n"
            "- Preserve numbers, percentages, ranges (e.g. 4-6), and symbols as-is.\n"
            "- Translate the ENTIRE text. Do not leave any $lang_in words or "
            "characters untranslated, including at the start or end of a fragment, "
            "even if the fragment looks incomplete.\n"
            "- Translate ONLY what is written. Do NOT complete, continue, expand, or "
            "add content. If the source is a truncated fragment (e.g. ends with '…' "
            "or cuts off mid-word), translate just that fragment and keep it about the "
            "same length; never invent the missing continuation.\n"
            "- Output only the translated text, nothing else.\n\n"
            "Source Text: $text\n\nTranslated Text:"
        )

    hl.translate(
        files=[args.input],
        output=output_dir,
        lang_in=args.lang_in,
        lang_out=args.lang_out,
        service=args.service,
        thread=args.thread,
        model=OnnxModel.load_available(),
        ignore_cache=args.fresh,
        prompt=llm_prompt,
    )

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_paths = {}
    for suffix, label in (("mono", "纯翻译版"), ("dual", "双语对照版")):
        path = os.path.join(output_dir, f"{base}-{suffix}.pdf")
        if os.path.exists(path):
            out_paths[suffix] = path
            size = os.path.getsize(path) // 1024
            print(f"✅ {label}: {path} ({size} KB)")

    # 图像内文字翻译(后处理)：翻译图表位图中烧录的源语言文字。
    if args.translate_images and not args.no_patch:
        try:
            from pdf2zh_patch.image_translate import translate_images_in_pdf
        except Exception as e:
            print(f"⏭️  图像翻译模块加载失败，已跳过: {e}")
            translate_images_in_pdf = None
        if translate_images_in_pdf is not None:
            chosen_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            for suffix in ("mono", "dual"):
                if suffix not in out_paths:
                    continue
                print(f"🖼️  翻译图像内文字: {os.path.basename(out_paths[suffix])} …")
                imgs, blocks = translate_images_in_pdf(
                    out_paths[suffix], args.lang_in, args.lang_out,
                    args.service, chosen_model)
                if blocks:
                    print(f"   ✅ 替换了 {blocks} 处图像文字(涉及 {imgs} 张图)")
                else:
                    print("   (未发现可翻译的图像文字)")

    # 漏译提示：当目标语言非中日韩时，扫描输出 PDF，列出仍残留中日韩字符的文字
    # (无论来自翻译漏译还是排版残留)，便于人工复核。图表内的位图文字无法检出，
    # 仅检查 PDF 文本层。
    _cjk_targets = {"zh", "zh-cn", "zh-tw", "ja", "ko", "chinese", "japanese", "korean"}
    if args.lang_out.lower() not in _cjk_targets:
        _scan_leftover_cjk(os.path.join(output_dir, f"{base}-mono.pdf"))

    return 0


def _scan_leftover_cjk(mono_path):
    """扫描译文 PDF 文本层，提示仍含中日韩字符的片段(可能需人工复核)。"""
    if not os.path.exists(mono_path):
        return
    try:
        import re
        import fitz  # PyMuPDF
    except Exception:
        return
    cjk = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
    # LLM 残留：提示词回显 / 拒绝语 — 不应出现在正常译文里。
    artifact = re.compile(
        r"^(translated\s+text|translation|target\s+text)\s*[:：]\s*$"
        r"|provide the source|no source text|i'?m sorry|as an ai",
        re.IGNORECASE,
    )
    found = []
    try:
        doc = fitz.open(mono_path)
        for pid in range(doc.page_count):
            for blk in doc[pid].get_text("dict")["blocks"]:
                for line in blk.get("lines", []):
                    for sp in line.get("spans", []):
                        t = sp["text"].strip()
                        if cjk.search(t):
                            found.append((pid + 1, "未译", t[:50]))
                        elif artifact.search(t):
                            found.append((pid + 1, "残留", t[:50]))
        doc.close()
    except Exception:
        return
    if found:
        print(f"\n⚠️  输出中有 {len(found)} 处疑似问题文字(未译/LLM 残留)，建议人工复核：")
        for pg, kind, txt in found[:20]:
            print(f"   • 第 {pg} 页 [{kind}]: {txt!r}")
        if len(found) > 20:
            print(f"   …… 其余 {len(found) - 20} 处略")


if __name__ == "__main__":
    sys.exit(main())
