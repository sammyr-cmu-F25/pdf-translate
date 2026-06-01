# pdf-translate-skill

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) custom skill that translates PDF files while preserving the original layout, vector graphics, tables, and formatting.

Powered by [PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate).

## What it does

- Translates PDF text content to your target language
- Preserves the original two-column layout, tables, charts, and vector graphics
- Outputs both a translated-only version (`-mono.pdf`) and a bilingual version (`-dual.pdf`)
- Supports multiple translation services (Google, DeepL, OpenAI, Ollama)

## Installation

### Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (CLI or desktop app)
- Python 3.10+

### Install the skill

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/pdf-translate-skill.git

# Copy the skill files to Claude Code skills directory
# macOS / Linux
cp pdf-translate-skill/pdf-translate.md ~/.claude/skills/
cp pdf-translate-skill/pdf-translate.py ~/.claude/skills/

# Windows (PowerShell)
Copy-Item pdf-translate-skill\pdf-translate.md $env:USERPROFILE\.claude\skills\
Copy-Item pdf-translate-skill\pdf-translate.py $env:USERPROFILE\.claude\skills\
```

### Install pdf2zh

```bash
pip install pdf2zh
```

## Usage

In a Claude Code session, type:

```
/pdf-translate
```

Claude will ask you for:
1. **PDF file path** (required)
2. **Source language** (default: `en`)
3. **Target language** (default: `zh`)
4. **Translation service** (default: `google`, free)

Or provide everything at once:

```
/pdf-translate /path/to/document.pdf en zh google
```

### Standalone script

You can also run the Python script directly:

```bash
python pdf-translate.py input.pdf                    # English → Chinese
python pdf-translate.py input.pdf -li ja -lo zh      # Japanese → Chinese
python pdf-translate.py input.pdf --service deepl     # Use DeepL
```

## Supported languages

| Language   | Code | Language   | Code |
|-----------|------|-----------|------|
| English   | en   | Chinese   | zh   |
| Japanese  | ja   | Korean    | ko   |
| French    | fr   | German    | de   |
| Spanish   | es   | Russian   | ru   |

## Supported translation services

| Service        | Flag                  | API Key Required | Quality |
|---------------|-----------------------|-----------------|---------|
| Google Translate | `--service google`  | No (free)       | ★★★   |
| DeepL          | `--service deepl`     | `DEEPL_AUTH_KEY` | ★★★★  |
| OpenAI         | `--service openai`    | `OPENAI_API_KEY` | ★★★★  |
| Ollama (local) | `--service ollama`    | No              | ★★★   |

## How it works

1. Extracts text spans and their exact coordinates from the PDF
2. Calls the translation API to translate each text span
3. Replaces the original text at the same position with the translated version
4. All vector graphics, lines, images, and decorative elements are preserved

## Known limitations

- Scanned PDFs (pure image-based) are not supported — use OCR first
- Machine translation may need manual review for domain-specific terminology
- Large files (>50 pages) may take a while

## License

MIT
