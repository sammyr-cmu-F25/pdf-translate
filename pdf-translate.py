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

    parser = argparse.ArgumentParser(description="Translate a PDF while preserving layout (patched pdf2zh).")
    parser.add_argument("input", help="Path to the input PDF")
    parser.add_argument("-li", "--lang-in", default="en", help="Source language code (default: en)")
    parser.add_argument("-lo", "--lang-out", default="zh", help="Target language code (default: zh)")
    parser.add_argument("-s", "--service", default="google", help="Translation service (default: google)")
    parser.add_argument("-o", "--output", default="", help="Output directory (default: alongside input)")
    parser.add_argument("-t", "--thread", type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument("--no-patch", action="store_true", help="Disable enhancement patches (use stock pdf2zh)")
    parser.add_argument("--protect-figures", action="store_true",
                        help="Do NOT translate text inside figures/tables (restore stock pdf2zh behavior). "
                             "By default this tool translates that text too (math formulas always preserved).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return 1

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

    hl.translate(
        files=[args.input],
        output=output_dir,
        lang_in=args.lang_in,
        lang_out=args.lang_out,
        service=args.service,
        thread=args.thread,
        model=OnnxModel.load_available(),
    )

    base = os.path.splitext(os.path.basename(args.input))[0]
    for suffix, label in (("mono", "纯翻译版"), ("dual", "双语对照版")):
        path = os.path.join(output_dir, f"{base}-{suffix}.pdf")
        if os.path.exists(path):
            size = os.path.getsize(path) // 1024
            print(f"✅ {label}: {path} ({size} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
