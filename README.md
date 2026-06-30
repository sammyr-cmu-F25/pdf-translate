# pdf-translate

Translate a PDF into another language while **preserving the original layout** —
two-column text, tables, charts, vector graphics, and formatting all stay in
place. Outputs a translated-only PDF and a bilingual (side-by-side) PDF.

Built on [PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate)
with extra fixes for color matching, adaptive font sizing, CJK line wrapping,
rotated/landscape tables, and translating text baked into charts/figures.

---

## One-line usage

From a terminal, point the `translate` command at any PDF:

```bash
# English → Chinese (uses OpenAI / GPT-4o — best quality)
./translate ~/Downloads/paper.pdf -li en -lo zh --service openai

# Chinese → English
./translate ~/Downloads/报告.pdf -li zh -lo en --service openai

# Free, no API key (Google Translate)
./translate ~/Downloads/paper.pdf -li en -lo zh
```

Outputs land next to the input (or in `-o <dir>`):

- `paper-mono.pdf`  — translated only
- `paper-dual.pdf`  — original + translation side by side

The `./translate` wrapper uses the project's bundled virtualenv automatically —
**no `source .venv/bin/activate` needed.**

### Run it from anywhere (optional)

Add it to your PATH once, then call `translate` from any folder:

```bash
ln -s "$PWD/translate" /usr/local/bin/translate     # run this from the project dir
translate ~/Downloads/paper.pdf -li en -lo zh --service openai
```

---

## Common options

| Option | Meaning | Default |
|--------|---------|---------|
| `-li`  | source language (`en`, `zh`, `ja`, `ko`, …) | `en` |
| `-lo`  | target language | `zh` |
| `--service` | `google` (free), `openai`, `deepl`, `ollama` | `google` |
| `-m, --model` | LLM model (OpenAI) | `gpt-4o` |
| `-o`   | output directory | next to input |
| `--translate-images` | also translate text inside charts/figures (see below) | off |
| `--fresh` | ignore the translation cache, re-translate everything | off |
| `-t`   | worker threads | `4` |

### Translation services

| Service | Flag | API key | Quality |
|---------|------|---------|---------|
| Google  | `--service google` | none (free) | ★★★ |
| OpenAI  | `--service openai` | `OPENAI_API_KEY` | ★★★★ |
| DeepL   | `--service deepl`  | `DEEPL_AUTH_KEY` | ★★★★ |
| Ollama  | `--service ollama` | none (local) | ★★★ |

Set your OpenAI key once (zsh):

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc && source ~/.zshrc
```

---

## Translating text inside charts / figures

Chart titles, axis labels, and legends are often **not real text** — they're
baked into a bitmap or drawn as vector outlines, so the normal pipeline can't
reach them. Add `--translate-images`:

```bash
./translate ~/Downloads/report.pdf -li zh -lo en --service openai --translate-images
```

It detects text in figures (EasyOCR), translates it (GPT-4o vision), and redraws
it in place — handling **bitmap charts, vector-drawn labels, and rotated /
landscape tables**, in both directions (CJK ↔ English).

Requires `--service openai` and a one-time `pip install easyocr` (into the
project venv: `.venv/bin/pip install easyocr`). It adds an OCR pass plus a few
vision calls per figure, so it's slower than text-only translation.

---

## First-time setup

Already set up in this folder (`.venv/` exists). To recreate it elsewhere:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# optional, for --translate-images:
.venv/bin/pip install easyocr
```

Requires Python 3.10–3.12 (pdf2zh is not compatible with 3.13).

---

## Known limitations

- Scanned PDFs (pure images, no text layer) need OCR first — not handled here.
- Machine translation may need manual review for domain terminology.
- `--translate-images`: redrawn font is a system default (not the chart's
  original), very dense legends may clip slightly, and it's slower/costs vision
  calls. After a run, any text that still couldn't be translated is flagged in
  the terminal with page numbers.

## License

MIT — Copyright (c) 2026 [银桑](https://github.com/yinsang0910-star)
