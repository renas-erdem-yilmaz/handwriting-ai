# Handwriting AI 1.0 — local notes to handwritten PDF

The Streamlit application reuses the existing DiffusionPen/Reforge inference,
candidate cache/reviewer, x-height normalization, page layout and approved-symbol
math renderer. It does not train models or require mathematical symbol collection.
The existing Tkinter and command-line applications remain available.

## First run on Windows / PowerShell

Run from the project root:

```powershell
# Install only the application additions in the existing Python 3.11 environment.
.\.reforge\Scripts\python.exe -m pip install -r requirements-app.txt

# Launch on this computer only.
.\.reforge\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**. The project config disables usage telemetry and
file watching of the large model/data trees. Nothing is deployed publicly.
Optional PowerShell activation: `.\.reforge\Scripts\Activate.ps1`; direct
interpreter commands above work without activation. Python 3.11, the model stack,
cached weights and existing datasets must already be installed as in Steps 5–9.
The new Streamlit package and its dependencies were installed in `.reforge`; no
existing inference packages were upgraded and `.venv` was left unchanged.

1. Start with **Existing Erdem style**, or select **Choose five words**. Choose
   exactly five different prepared word images and inspect the reference previews.
   Existing annotations label these choices; no symbol labels are inferred.
   A custom profile is stored privately without changing any original images.
2. Use the sample or paste notes in **Editor**. Blank lines separate paragraphs;
   single newlines within a paragraph wrap naturally. `# Motion notes` is a heading.
3. Choose A4/notebook, blank/lined background, writing x-height, spacing, individual
   margins, darkness and natural variation. Advanced controls include seed, sampling
   steps, device and initial candidate count. Pretrained is the default; the pilot
   adapter is optional and has no established quality improvement.
4. Click **Generate handwriting**. One shared background worker owns the model;
   progress polls without freezing the editor. No inference runs on startup,
   switching tabs, changing controls, enlarging previews or downloading files.
5. Inspect **Preview & export**. Every page is shown. Use Fit/Enlarge/Full resolution
   and download individual PNGs or the multipage PDF. Changed controls do not replace
   the displayed snapshot until Generate is pressed. Downloads match that snapshot.
6. In **Word review**, inspect all words or filter structural flags. A lack of flags
   is not evidence of correct spelling. Select a candidate and click **Use and
   approve**, **Use without approval**, or **Mark problematic**. Choices persist in
   the existing review store. **Generate another candidate** appends one new seed
   for that word, leaving previous decisions and unrelated words intact.
7. Selecting a candidate refreshes the complete layout from cached words without
   inferring other words. Complete-page recomposition is necessary because a word's
   changed width can alter following line breaks/page boundaries. Download the
   refreshed PDF when it is ready.

Cancel is checked between word operations; an active CUDA call/model load can finish
before cancellation takes effect. Finished candidates remain cached. Multiple browser
sessions share one worker: if another job is running, wait and retry. Avoid running
other inference/regeneration programs concurrently. Streamlit session state retains
notes/results/selections across reruns, not a completely closed browser/server restart.
Every completed run is saved privately for reproducibility and later CLI review.

## Input and mathematics

Prose follows the existing supported English vocabulary: ASCII letters, internal
apostrophes/hyphens and one trailing punctuation mark `.,;:!?`. Unsupported words,
unmatched delimiters and invalid settings produce explicit errors before inference;
they are never silently dropped. Place numbers and symbols inside math delimiters.

Use `$...$` inline, or a separate `$...$` / `$$...$$` paragraph for a displayed
equation. Examples:

```text
# Physics

A net force changes the motion of a body.

$F = ma$

$a = F/m$

$v = v_0 + at$

$x = 1/2 at^2$

$θ = α + β$

$v^2 = v_0^2 + 2ax$
```

This is the existing restricted syntax: single-letter variables, arithmetic,
parentheses, slash fractions and `^`/`_` scripts. Use `x^{12}` for compound powers,
`F/(ma)` for grouped denominators; `F/ma` means `(F/m) × a`. Existing Mathtext
commands retain the original font fallback. See
[symbol syntax and collection guide](page_engine/math_symbols/README.md).

When every required symbol has an explicitly approved crop, the handwritten
symbol assembler is used. Otherwise the whole equation uses labeled standard-font
fallback. Preview warnings, page footers and manifests record provenance and missing
inventory. This works before you collect symbols. Open the existing local collector:

```powershell
.\.reforge\Scripts\python.exe -m page_engine.math_symbols collect
```

After collecting/reviewing/approving symbols, **Refresh pages from saved word
choices / math library** to use them. The web application never auto-approves symbols.

## Backend and preservation

| Feature | Reused implementation | Current status |
|---|---|---|
| Pretrained model + shared five-reference style | `page_engine/model.py` | Fresh CUDA inference verified; lazy strict checkpoint loading |
| Optional style projection adapter | Same pipeline, existing `last_adapter.pt` | Fresh CUDA inference via Streamlit verified; checkpoint unchanged; no improvement claim |
| Word cache, candidates, persistent decisions | `page_engine/words.py`, `candidates.py` | Generation, additional candidates and isolated approval tests passed |
| Ink/x-height/baseline normalization | `consistency.py`, `writing_scale.py` | Existing geometry and regression checks retained |
| Width-based wrapping and pagination | `page_engine/layout.py` | A4/notebook, multipage, bounds and overlap checks passed |
| Mathematical handwriting | `page_engine/math_symbols/` | Assembly tested with synthetic fixtures; real personalization awaits approved crops |
| PNG/PDF export | Existing `handwriting_app.service.export_pdf`, PNG copy helper | Lossless pixels, page order/size and PNG DPI verified |

The interface is `app.py` + `app_components/` (editor/settings/preview/review/export,
cached resources and a serialized worker). The single generation configuration is
the extended `handwriting_app.service.Settings`. Its defaults preserve the earlier
Tkinter/CLI behavior; additions expose notebook size, style profile, darkness and
bounded size variation. No model inference or math parsing architecture was replaced.

`st.cache_resource` retains one lazy backend/model owner, and a polling fragment
updates job progress. These follow Streamlit's documented
[resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
and [fragment](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)
interfaces. GPU model copies are not cached separately per settings combination.

New private run artifacts live under `outputs/streamlit_runs/`, caches/reviews/style
profiles under `.reforge-cache/`. Existing `.gitignore` covers these, datasets and
checkpoints. Original PDFs, data, annotations, weights, prior approvals, training
experiments and comparison images are preserved. Tests never install synthetic
images or approvals into the user's cache/library. Model/data are not bundled in Git.

Errors in the interface provide recovery instructions without displaying personal
absolute paths. Full tracebacks are saved with the failed local run. CUDA OOM recovery:
close other GPU jobs or choose CPU and retry. Missing checkpoint/reference recovery:
restore the existing files/configuration; do not retrain to fix installation errors.

## Verification and samples

```powershell
.\.reforge\Scripts\python.exe -m unittest app_components.tests handwriting_app.tests page_engine.tests page_engine.quality_tests page_engine.scale_tests page_engine.math_symbols.tests
```

Test results: `outputs/app_tests/streamlit_v1/`. The real smoke test exercised actual
Streamlit controls with the pretrained model on CUDA (fresh inference and a second
word candidate), pending selection, cached notebook/A4 generation, PNG downloads
and PDFs. Generation is not mocked in that smoke test. Approval persistence tests
use isolated synthetic fixtures to avoid fabricating human approval of real outputs.

Samples: `a4/document.pdf` (one page), `notebook/document.pdf` (two pages, deliberately
wide 18 mm line pitch), and `real_word/document.pdf`. The real-word candidate shows
inherited model letter distortions and remains pending. Reports include hardware,
cache/model identity, original-file hashes and pixel comparisons. PDF pages were
rendered and visually checked. Automated browser capture was unavailable because
its sandbox helper failed; localhost HTTP startup and actual Streamlit AppTest
interactions were verified. This is not a claim of a completed screenshot audit.

The real smoke module is `python -m app_components.smoke`; its existing outputs are
preserved and repeat execution refuses to overwrite the report. Ordinary Generate
and the existing CLI tools create unique run directories instead.

## Limitations and useful next improvements

DiffusionPen can distort/misspell words, merge letters, or alternate between cursive
and printed forms. Candidates/retries cannot guarantee spelling or personal style
similarity. Review the output. The pilot adapter's tiny training experiment did not
establish meaningful improvement. Ink/x-height estimates remain heuristic.

PDFs contain raster handwriting, not editable/searchable text. Large uncached
documents may take hours; v1 accepts at most 100,000 characters. Safe serial model
inference is retained, with word caching; unverified batching was not added.
Full-page recomposition can change pagination after a word replacement. PNG DPI is
encoded to the nearest supported pixels-per-meter value (~149.987 for 150 DPI).

Useful post-v1 work: a saved-document reopen browser, richer zoom/crop inspection,
optional OCR-assisted review, a larger reviewed word/symbol dataset, and measured
performance improvements. No extra retraining or public deployment was performed.
