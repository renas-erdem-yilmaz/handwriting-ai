# Step 10 implementation report

Implemented `handwriting_app/`: GUI, reusable service, CLI, tests, real-inference
smoke test, example notes, dependency pins and usage documentation. The only
Step 1–9 integration change is `page_engine/layout.py`: Letter page size and a
blank/lined background setting independent of page size. Existing notebook/A4
defaults, normalization, symbol review, inference and checkpoint loaders remain.

The app provides UTF-8 text loading, multi-paragraph editing/headings, A4/Letter,
four margins, line/paragraph/word spacing, writing x-height, seed, candidate count,
device and optional existing adapter. One background worker owns inference;
Tk receives queued progress/results/errors. Cancellation stops between words.
Preview navigation includes Fit/75/100/150% zoom and scrolling. Input edits disable
export until regenerated. Each job saves input/settings/pages, source provenance,
cache IDs, review session, page hashes and diagnostic logs on errors.

Export embeds original preview RGB pixels losslessly into correct-size PDF pages
in order. It checks input token count/order, page bounds/overlap and source hashes,
validates PDF page count, and refuses existing filenames/protected destinations.
Mathematical notation uses the existing approved-symbol renderer with explicitly
labeled font fallback. No symbol collection is required to generate/export notes.

## Actual validation

- 26 application/regression tests passed (8 application tests + 18 prior tests).
  Coverage includes long paragraphs, multi-page Letter/lined layout, empty/invalid
  input, no-symbol fractions/scripts, missing-model recovery, cancellation, real
  Tk background-worker responsiveness, navigation, zoom and export, and protected
  output paths. PDF tests compare decoded embedded RGB bytes with every page PNG.
- Fresh original pretrained CUDA inference generated `Physics` and 32 prose words
  for the physics sample; no model training occurred. Peak allocated GPU memory
  was 1.047 GiB. Initial model loading plus generation/validation/export took
  209.7 seconds. The exact pretrained/style/reference identities are recorded in
  `outputs/app_tests/integration_report.json`.
- A normal CLI rerun reused all 32 words, with zero new candidates, and
  produced a default-spacing Letter/lined sample in about four seconds. Its
  manifest is under `outputs/app_tests/cache_verified/`.
- `outputs/app_tests/handwritten_notes.pdf` is a 2-page A4 pagination demonstration
  using intentionally wide 18 mm line spacing. `notes_default.pdf` is a 1-page
  Letter/lined sample with ordinary 8 mm spacing. `real_inference.pdf` records the
  single-word fresh-inference smoke test. PDFs were reopened/rendered with the
  original project's existing PyMuPDF environment and visually inspected; equations,
  page order, margins and fallback labels are intact. The second demonstration
  page contains the final equation, without lost input or overlapping content.
- 1,056 recorded original datasets/outputs remained unchanged. Pretrained weights,
  style encoder, reference images and adapter checkpoints were preserved.
- Installed only `reportlab==5.0.1` and `pypdf==6.20.0` in `.reforge` for PDF
  creation/validation; the main `.venv` was not changed. Original PyMuPDF is reused
  for visual PDF QA, and is not a new application dependency.

## Remaining limits and setup

Launch: `.\.reforge\Scripts\python.exe -m handwriting_app` from the project root.
No manual model or math setup is required in this existing environment. Collecting
approved symbols later enables personal mathematics automatically for complete
symbol inventories. The current equations remain visibly standard-font fallback.

DiffusionPen spelling/shape issues remain: `acceleration.` and `same` received
structural edge flags in the sample. Such flags appear in the preview panel and
manifest; an unflagged word is not necessarily correct. All newly generated words
remain pending human review. The pilot adapter stays optional without any quality
improvement claim. Review candidates using the existing tool and regenerate with
the same settings to reuse your choices.

PDFs contain raster handwriting, not editable/searchable text. First-time long
documents may take hours; CPU inference is slower. The worker cannot interrupt a
CUDA call or model loading midway. Safe serial inference reuses the model and
caches; this implementation does not add unverified batched model inference.
