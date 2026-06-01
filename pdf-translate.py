#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 翻译工具 — 基于 pdf2zh (PDFMathTranslate)
保留原始排版、图形、表格，仅替换文本为目标语言

用法:
    python pdf-translate.py input.pdf                    # 默认英文→中文
    python pdf-translate.py input.pdf -li ja -lo zh      # 日文→中文
    python pdf-translate.py input.pdf --service deepl     # 指定翻译服务

依赖:
    Python 3.10+
    pip install pdf2zh

输出:
    input-mono.pdf  — 纯目标语言版
    input-dual.pdf  — 双语对照版
"""

import subprocess
import sys
import os
import shutil


def find_pdf2zh():
    """查找 pdf2zh 可执行文件路径"""
    # 优先查找 PATH 中的 pdf2zh
    pdf2zh = shutil.which("pdf2zh")
    if pdf2zh:
        return pdf2zh

    # 查找 Scripts/bin 目录下的 pdf2zh.exe
    python_dir = os.path.dirname(sys.executable)
    for subdir in ("Scripts", "bin"):
        candidate = os.path.join(python_dir, subdir, "pdf2zh.exe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(python_dir, subdir, "pdf2zh")
        if os.path.isfile(candidate):
            return candidate

    # 兜底：用 python -m pdf2zh
    return None


def translate_pdf(input_pdf, lang_in="en", lang_out="zh", service="google"):
    """翻译 PDF 并保留原始排版"""
    if not os.path.exists(input_pdf):
        print(f"❌ 文件不存在: {input_pdf}")
        return None

    pdf2zh_cmd = find_pdf2zh()
    if pdf2zh_cmd:
        cmd = [pdf2zh_cmd, input_pdf, "-li", lang_in, "-lo", lang_out, "--service", service]
    else:
        cmd = [sys.executable, "-m", "pdf2zh", input_pdf, "-li", lang_in, "-lo", lang_out, "--service", service]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    print(f"🔄 翻译中: {os.path.basename(input_pdf)}")
    print(f"   {lang_in} → {lang_out} (服务: {service})")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

    if result.returncode != 0:
        print(f"❌ 翻译失败:\n{result.stderr}")
        return None

    # 确定输出文件名
    base = os.path.splitext(input_pdf)[0]
    mono = f"{base}-mono.pdf"
    dual = f"{base}-dual.pdf"

    outputs = {}
    if os.path.exists(mono):
        size = os.path.getsize(mono) // 1024
        print(f"✅ 纯翻译版: {mono} ({size} KB)")
        outputs["mono"] = mono
    if os.path.exists(dual):
        size = os.path.getsize(dual) // 1024
        print(f"✅ 双语对照版: {dual} ({size} KB)")
        outputs["dual"] = dual

    return outputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python pdf-translate.py <input.pdf> [-li en] [-lo zh] [--service google]")
        sys.exit(1)

    input_file = sys.argv[1]

    lang_in = "en"
    lang_out = "zh"
    service = "google"

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "-li" and i + 1 < len(args):
            lang_in = args[i + 1]; i += 2
        elif args[i] == "-lo" and i + 1 < len(args):
            lang_out = args[i + 1]; i += 2
        elif args[i] == "--service" and i + 1 < len(args):
            service = args[i + 1]; i += 2
        else:
            i += 1

    translate_pdf(input_file, lang_in, lang_out, service)
