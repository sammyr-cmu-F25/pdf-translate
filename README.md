# pdf-translate-skill

[English](#english) | [中文](#中文) | [日本語](#日本語)

---

## English

### Translate PDFs in Claude Code — Layout Preserved

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) custom skill that translates PDF files into another language while **preserving the original layout, vector graphics, tables, and formatting**.

Powered by [PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate).

**Features:**
- Preserves original two-column layout, tables, charts, and vector graphics
- Outputs both a translated-only version (`-mono.pdf`) and a bilingual version (`-dual.pdf`)
- Supports 8+ languages (English, Chinese, Japanese, Korean, French, German, Spanish, Russian)
- Supports multiple translation services: Google (free), DeepL, OpenAI, Ollama (local)
- Auto-detects and installs dependencies (Python 3.10+, pdf2zh)

**Quick Start:**

```bash
# 1. Install the skill
git clone https://github.com/yinsang0910-star/pdf-translate-skill.git
cp pdf-translate-skill/pdf-translate.md ~/.claude/skills/
cp pdf-translate-skill/pdf-translate.py ~/.claude/skills/

# 2. Install pdf2zh
pip install pdf2zh

# 3. Use in Claude Code
/pdf-translate /path/to/document.pdf
```

---

## 中文

### 在 Claude Code 中翻译 PDF —— 完整保留排版

一个 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 自定义技能，可将 PDF 文件翻译为另一种语言，同时**完整保留原始排版、矢量图形、表格和格式**。

基于 [PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate)。

**功能特性：**
- 保留原始两栏布局、表格、图表和矢量图形
- 同时输出纯翻译版（`-mono.pdf`）和双语对照版（`-dual.pdf`）
- 支持 8+ 种语言（英语、中文、日语、韩语、法语、德语、西班牙语、俄语）
- 支持多种翻译服务：Google（免费）、DeepL、OpenAI、Ollama（本地）
- 自动检测并安装依赖（Python 3.10+、pdf2zh）

**快速开始：**

```bash
# 1. 安装技能
git clone https://github.com/yinsang0910-star/pdf-translate-skill.git
cp pdf-translate-skill/pdf-translate.md ~/.claude/skills/
cp pdf-translate-skill/pdf-translate.py ~/.claude/skills/

# 2. 安装 pdf2zh
pip install pdf2zh

# 3. 在 Claude Code 中使用
/pdf-translate /path/to/document.pdf
```

---

## 日本語

### Claude Code で PDF を翻訳 — レイアウトを保持

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) のカスタムスキル。PDFファイルを別言語に翻訳し、**元のレイアウト・ベクトルグラフィック・表・フォーマットをそのまま保持**します。

[PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate) により動作します。

**機能：**
- 元の2段組みレイアウト・表・チャート・ベクトルグラフィックを保持
- 翻訳のみ版（`-mono.pdf`）とバイリンガル版（`-dual.pdf`）を同時出力
- 8以上の言語対応（英語、中国語、日本語、韓国語、フランス語、ドイツ語、スペイン語、ロシア語）
- 複数の翻訳サービスに対応：Google（無料）、DeepL、OpenAI、Ollama（ローカル）
- 依存関係の自動検出・インストール（Python 3.10+、pdf2zh）

**クイックスタート：**

```bash
# 1. スキルをインストール
git clone https://github.com/yinsang0910-star/pdf-translate-skill.git
cp pdf-translate-skill/pdf-translate.md ~/.claude/skills/
cp pdf-translate-skill/pdf-translate.py ~/.claude/skills/

# 2. pdf2zh をインストール
pip install pdf2zh

# 3. Claude Code で使用
/pdf-translate /path/to/document.pdf
```

---

## Supported Translation Services / 支持的翻译服务 / 対応翻訳サービス

| Service | Flag | API Key | Quality |
|---------|------|---------|---------|
| Google Translate | `--service google` | No (free) | ★★★ |
| DeepL | `--service deepl` | `DEEPL_AUTH_KEY` | ★★★★ |
| OpenAI | `--service openai` | `OPENAI_API_KEY` | ★★★★ |
| Ollama (local) | `--service ollama` | No | ★★★ |

## How It Works / 工作原理 / 仕組み

1. Extract text spans and their exact coordinates from the PDF
2. Call the translation API to translate each text span
3. Replace original text at the same position with translated version
4. All vector graphics, lines, images, and decorative elements are preserved

PDFのテキストスパンと正確な座標を抽出 → 翻訳APIを呼び出して翻訳 → 同じ位置に翻訳テキストを配置 → ベクトルグラフィック・線・画像はそのまま保持

## Translating text baked into charts/images / 翻译图表位图文字

Chart titles, axis labels, and legends are often rasterized into the figure
bitmap (no text layer), so the normal pipeline can't reach them. Pass
`--translate-images` to also translate that text via OCR + redraw:

```bash
python pdf-translate.py input.pdf -li zh -lo en --service openai --translate-images
```

Requirements: `--service openai` (uses GPT-4o vision) and `pip install easyocr`.
Works **both directions** — CJK↔English (zh/ja/ko ↔ en); source and target must
differ in script. It detects text regions with EasyOCR, transcribes+translates
each with the vision model, erases the original pixels, and redraws the
translation (color/size matched, CJK-capable font when the target is Chinese/
Japanese/Korean), preserving the image's transparency. Cost: an OCR model load
plus one vision call per detected label.

## Known Limitations / 已知限制 / 既知の制限

- Scanned PDFs (pure image-based) are not supported — use OCR first
- Machine translation may need manual review for domain-specific terminology
- Large files (>50 pages) may take longer
- `--translate-images` covers CJK↔English; redrawn font is a system default
  (not the chart's original font), and very dense legends may clip slightly

## License

MIT - Copyright (c) 2026 [银桑](https://github.com/yinsang0910-star)
