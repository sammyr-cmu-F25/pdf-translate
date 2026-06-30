

# Research: Translating Chinese text baked into chart/figure images

Status: research notes (2026-06-29). Not yet implemented in the pipeline.

## Problem

In the source PDFs, charts (bar/line/bubble) are embedded as **single rasterized
bitmap images**, not vector text. Examples from `H3_AP202404121630191621_1.pdf`:

- Page 10 bar chart: one `741×320` ICC image (xref 95).
- Page 14 bubble chart: one `789×560` ICC image (xref 107).

The chart **title, axis labels, and legends are baked into the pixels** (e.g.
title "小分子药物行业规模", y-axis "亿/人民币元" and "百分比"). There is no
text layer or vector path for them, so the normal text-substitution pipeline
(which rewrites the PDF text layer) cannot touch them. This is the one class of
text the tool leaves untranslated.

## What was tested

Environment: PIL 11, numpy 2.4, OpenCV 4.13, OpenAI SDK 2.44 (vision). No
`tesseract` binary. (PaddleOCR/EasyOCR installation attempted — heavy deps.)

### Finding 1 — Whole-image vision OCR confabulates
Sending the full `741×320` chart to GPT-4o ("transcribe the Chinese") returned
**wrong** text every time, pattern-matched from layout priors:
- real title "小分子药物行业规模" → got "小区平均房价趋势", "小区历年成交量",
  even "小心地滑注意安全".
The red/green dual-axis layout makes it "guess" real-estate / safety-sign
phrases instead of reading glyphs. Contrast boosting did not help.

### Finding 2 — Tight high-res crops OCR correctly
Render the chart region from the PDF **page** at 600 DPI
(`page.get_image_rects(xref)` → `get_pixmap(matrix, clip=rect)`), composite on
white, crop to a SINGLE text region, upscale 2× LANCZOS, then OCR that crop:
- title crop → "小分子药物行业规模" ✓ (correct)
- axis crop → "亿/人民币元" and "百分比" ✓ (correct)

So vision OCR accuracy is gated on **crop tightness + resolution**, not the
model.

### Finding 3 — All-vision two-pass (detect then OCR) is fragile
Asking GPT-4o to first return text-region bounding boxes, then OCR each crop:
the **detection bboxes were imprecise**, so crops missed/clipped the text and
the OCR step confabulated again ("对不起，我爱你"). Vision-model bbox precision
is not good enough to drive the crops.

### Finding 4 — Hybrid EasyOCR(boxes) + GPT-4o(recognize crop) WORKS ✓
EasyOCR (`ch_sim`+`en`, CPU) on the hi-DPI chart gives **accurate bounding
boxes** but mediocre recognition on small text (title → "小子药物行业规模",
missing a char; axis → "亿 ^民币示"). GPT-4o has the opposite profile. Combine
them: take each EasyOCR CJK box → crop (+pad) → upscale 2× → GPT-4o transcribe
+translate the tight crop. Validated end-to-end on page-10 bar chart:
- box "小子药物行业规模" → **小分子药物行业规模** | Small molecule … industry scale ✓
- box "亿 ^民币示"      → **亿人民币元** | Billion RMB Yuan ✓
- box "巨"             → **百分比** | Percentage ✓
All three correct. EasyOCR supplies the boxes vision lacks; GPT-4o fixes the
recognition EasyOCR gets wrong. This is the detect+recognize+translate chain.

## Recommended pipeline (if pursued)

Detect+recognize+translate is **validated** (Finding 4). Full pipeline:

1. For each figure image: get its page rect, render that region at high DPI from
   the page (sharp, white background). `page.get_image_rects(xref)` →
   `get_pixmap(matrix=600/72, clip=rect)`.
2. **Detect** text boxes with EasyOCR (`ch_sim`+`en`, CPU). Keep boxes whose
   rough text contains CJK. (Reliable boxes; recognition often wrong on small
   text — that's fine.)
3. **Recognize+translate per box**: crop (+~8px pad), upscale 2× LANCZOS, send
   to GPT-4o "transcribe exactly, then translate". This corrects EasyOCR's
   recognition AND translates in one call.
4. **Erase** original text pixels (inpaint / fill with sampled background
   color) and **redraw** the English with PIL, matching color and fitting the
   box (shrink-to-fit like the text-layer PATCH#3/#5).  [NOT yet validated]
5. Re-embed the edited image back into the PDF at the same rect via
   `page.insert_image(rect, stream=...)` after deleting/covering the old one.
   [NOT yet validated]

### Finding 5 — Full pipeline (erase/redraw/re-embed) VALIDATED ✓
Proven end-to-end in the real PDF on the page-10 bar chart:
- Detect on the sharp hi-DPI page render; map each box back to embedded-image
  pixel coords by the linear scale (orig_w/hidpi_w, orig_h/hidpi_h).
- Recognize+translate each crop with GPT-4o.
- **Critical:** the embedded image is RGB with an OPAQUE BLACK background plus a
  SEPARATE soft-mask (smask) xref that supplies transparency. Extracting the RGB
  alone gives a black bg and confuses background sampling. Build the editable
  image as RGBA = `fitz.Pixmap(fitz.Pixmap(doc,xref), fitz.Pixmap(doc,smask))`
  so transparency is correct.
- Erase each text box to fully transparent `(0,0,0,0)`; pick text color = most
  saturated opaque pixel in the box; redraw English shrink-to-fit (<=1.5× box
  width, centered on box center, clamped to image).
- Re-embed with `page.replace_image(xref, stream=png_bytes)` — this replaces the
  pixels in place AND pymupdf re-derives the mask from the PNG alpha, so the
  transparent background is preserved. (Do NOT use `insert_image` overlay — it
  leaves the original image underneath, so both Chinese and English show.)

Result: chart renders with white (page) background, translated title
"Small molecule pharmaceutical industry scale" + axis "Billion RM"/"Percentage",
all bars/line/numbers intact, Chinese gone.

## Remaining polish (not blockers)
- Narrow axis-label boxes clip slightly ("Billion RM" vs "Billion RMB"); allow
  more width slack or abbreviate.
- Inpainting currently assumes text sits on flat/transparent bg; text overlapping
  bars/lines would need per-pixel background reconstruction (none in these
  charts, but possible elsewhere).
- Font is Arial; could match the chart's font family if detectable.
- Gate the whole feature behind a `--translate-images` flag (cost: EasyOCR load
  + one GPT-4o call per text region per figure).

## Caveats / open questions

- Font & color matching on redraw is approximate; charts use small bold fonts.
- Inpainting over text that overlaps bars/lines is hard; may need background
  sampling per-pixel or solid-fill heuristics.
- Cost: per-figure vision calls add up; gate behind a `--translate-images` flag.
- Some embedded images are low-res; upscaling can't recover lost detail.
- Numbers/years should be skipped (already numeric).

## Rotated text (landscape tables) — durable fix

Pages with landscape tables draw text rotated 90° (pdfminer line `dir=(0,-1)`,
char matrix `[0,b,c,0]`). pdf2zh's upright-only relayout turns this into
per-character vertical garbage (and misclassifies it as formula). Root-cause fix,
generalized so the whole class is handled:

1. **Converter (PATCH#8)**: detect chars whose matrix is rotated
   (`|m[0]|,|m[3]|≈0` and `|m[1]| or |m[2]|>0`); skip them entirely (don't
   translate, don't re-emit) and record their bboxes in
   `pdf2zh_patch.ROTATED_REGIONS[pageid]`. This alone removes the garbage.
2. **Image post-pass**: for each page, cluster the rotated char bboxes into
   blocks, rasterize each block from the ORIGINAL page, rotate to upright, OCR
   with EasyOCR, **batch-translate all cell texts in ONE LLM call**
   (`_batch_translate`, JSON list — NOT one vision call per cell, which timed
   out at 10 min), then draw the translations onto a tile, rotate back, and
   `insert_image` onto the output page. ~59s for this paper's 3 table pages.
   Applied to the **mono** output only (dual's two-column geometry differs).

Result: pages 4–6 landscape tables render as readable rotated Chinese instead of
scrambled vertical garbage.

## Vector-figure labels (no text layer, not a bitmap) — durable catch-all

Some figures (pie/scatter charts) draw their labels as **vector paths/outlined
glyphs**, so the text is neither in the PDF text layer NOR in an embedded bitmap.
Neither the text pipeline nor the image-xref pass touches them (e.g. Sakigake
paper page 7: 223 vector drawings, labels "Oncology/Neurology/..." absent from
text layer). General fix — a "translate uncovered text" pass:

- For pages with many vector drawings (gate: `get_drawings() >= 100`, skips
  text-only and watermark-only pages), rasterize the ORIGINAL page, OCR it.
- Drop any OCR box that overlaps a text-layer span (already translated).
- Batch-translate the remaining source-language boxes (one LLM call) and overlay
  each onto the output page (background-sampled fill + redrawn translation).
- Skip pages dominated by rotated text (`>=800` rotated chars → handled by the
  rotated pass); footer/sidebar watermark rotation (~hundreds) does NOT count.

This is the most general safety net: it translates page text regardless of how
it's drawn (vector, bitmap, or path), as long as the text layer doesn't already
cover it. Validated on Sakigake p7: pie/scatter labels → Chinese.

## Decision

Full pipeline is VALIDATED (Findings 1–5) and produces a correct translated
chart in the real PDF. To productionize: package as an opt-in `--translate-images`
flag (adds EasyOCR + torch deps and per-region GPT-4o calls). The text-layer
pipeline (committed in 7c2d8f1) already handles all real text; this extends
coverage to chart-baked text.

Prototype scripts live in /tmp during research; the productionized module would:
load EasyOCR once, iterate `page.get_images()` per page, run the 5 steps, and
`replace_image` each figure that contains CJK.
