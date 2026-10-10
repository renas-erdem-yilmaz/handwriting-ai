# Step 9.5 — consistency and candidate review

The subsequent word-size fix replaces the adaptive-v1 scale heuristic described
below with shared x-height normalization. See [SCALE.md](SCALE.md) for current
defaults, physical writing-size controls, confidence-based fallbacks and tests.
Earlier evaluation images and findings below describe their original implementation.

## Diagnosis

DiffusionPen conditions on the same five pooled style features but samples words
independently. This does not impose a hard writer-identity or common pen-width
constraint. Printed/cursive drift and inconsistent stroke geometry are model
limitations. Different random seeds can also omit/merge letters. Reforge stitches
long words and contractions; those boundaries can produce malformed letter groups.
Pixel processing cannot restore a missing letter or enforce correct spelling.

The old renderer scaled the entire ink box to one height (or 73% for words without
ascenders/descenders). Descenders and flourishes therefore changed body-letter size.
It removed pale background but left different ink opacities unchanged. The new path
uses actual column ink extents plus a conservative lexical body-height prior, caps
final heights to 70–118% of the configured height, and preserves aspect ratio. Its
baseline ignores terminal punctuation and estimates the body of descender words.
These are estimates, not a learned handwriting parser.

Adaptive opacity normalization uses the word's foreground quantile and shared
target opacity, with bounded gain. It removes only pale background texture. There
is no sharpening, thinning, erosion, dilation or stroke-width equalization. Thick
strokes and thin strokes can still differ; darkening thin letters does not make
their geometry match. Raw candidate images are never changed.

## Commands — PowerShell, main project directory

No new dependencies: existing Pillow, NumPy, PyTorch and Python's Tkinter are used.
Pretrained remains default; all checkpoint loads remain strict and local/offline.

```powershell
# Prepare a new document or resume interrupted candidate generation.
.\.reforge\Scripts\python.exe -m page_engine.quality prepare --document page_engine/examples/quality.json --session outputs/quality_tests/my_session.json

# Open the review window. Close any prepare/regenerate process before reviewing.
.\.reforge\Scripts\python.exe -m page_engine.quality review --session outputs/quality_tests/my_session.json

# Render using persistent selections; pending words use the first candidate.
.\.reforge\Scripts\python.exe -m page_engine.quality render --session outputs/quality_tests/my_session.json --output outputs/quality_tests/my_page

# Append a candidate for one occurrence; preserve its earlier selection and every other word.
.\.reforge\Scripts\python.exe -m page_engine.quality regenerate --session outputs/quality_tests/my_session.json --word-index 16 --seed 555555
```

To review the supplied generated experiment, use
`--session outputs/quality_tests/review_session.json` instead. No further model
loading occurs while merely reviewing or rendering prepared candidates. A GPU
model is loaded only for new inference. `--require-approved` on `render` refuses
unapproved/problematic words. Select a fresh output prefix each time; previous
previews/evaluations are preserved.

Prepare accepts `--candidates 1`, `2`, or `3` (default 3), `--steps` (default 20),
`--guidance` (default 3), `--ocr` (optional cached TrOCR), `--style` and
`--adapter outputs/finetune/erdem_pilot/last_adapter.pt`. Each saved candidate has
its own actual seed and image hash. There are no hidden best-only retries in this
path. Initial requests stop at three candidates; explicit regeneration appends
additional candidates. OCR mismatches are displayed, never treated as proof of
correctness or as writer-style scores.

The existing `python -m page_engine ...` Step 9 commands continue to work. Both
rendering entry points now default to adaptive x-height normalization; explicit
`processing.mode = "legacy"` retains the Step 9 bounding-box normalization.

## Shared style and review persistence

`styles/erdem.json` selects exactly five genuine training-only word crops and one
opacity target for the entire page. Copy this config to choose another five
references. Reference content hashes namespace the raw word cache and review
bundles; incompatible model/reference changes are rejected on resume. Session
files retain the full document, seed mapping, model hashes and sampling settings.
`word_seeds` may override individual occurrence seeds in document JSON.

Candidate bundles and decisions live in `.reforge-cache/candidate_reviews/`; raw
images remain in `.reforge-cache/words/`. Approval belongs to an exact candidate,
text and model/reference configuration. Normal reuse uses the same occurrence
seed and honors previous decisions, while different occurrence seeds preserve
variation. Opt-in `prepare --reuse-approved` can reuse a previously approved exact
word and its candidate pool across occurrence seeds without extra inference;
it explicitly records inherited approval and can make repeated words identical.
Leave it off when natural variation matters more than cross-occurrence reuse.

The Tkinter window displays raw candidates side by side. Double-click an image to
inspect it larger. Buttons save decisions immediately; navigation position also
persists. A background worker handles regeneration while Tk updates stay on the
main thread. Later-candidate buttons expose appended images. Mark problematic
when all candidates are unsuitable; the default preview still renders that word
with a manifest warning. It is not silently declared correct or discarded.
Run one writing process per review session at a time. Closing the window during
generation can interrupt a new candidate; completed decisions/candidates remain
and generation can be resumed. Use ordinary CLI regeneration if background
generation is inappropriate for your desktop.

## Layout configuration

`layout.word_spacing_mm`, `word_height_mm`, `line_spacing_mm`, margins,
paragraph spacing and indentation remain configurable. New `size_variation`
(0–0.05, default .05) and `vertical_variation_mm` (at most 3 pixels at configured
DPI) preserve bounded natural variation. New `processing.ink_strength` (0–1) and
`opacity_target` adjust tonal normalization. Mathematical blocks still use the
clearly labeled standard-font fallback.

## Evaluation

`quality_experiment.py` prepares the supplied experiment and a before/after pair
from exactly the same first candidates, document, style features and sampling.
Candidate 1 can reuse a Step 9 image only when its actual seed and model/reference
hashes match; an image selected by a different retry seed is not reused this way.
Thus this is an image-processing comparison, not evidence that model spelling
improved. Separate reviewed output can incorporate chosen alternatives; its
manifest records the exact candidate differences. The scan reference in the
comparison is genuine handwriting, labeled separately from generated output.

Run checks with:

```powershell
.\.reforge\Scripts\python.exe -m page_engine.tests
.\.reforge\Scripts\python.exe -m page_engine.quality_tests
```

No retraining, numerical writer-similarity score, guaranteed spelling, or final
application is included. Add more varied clean word references and have the writer
review samples before claiming personal-style fidelity. The existing tiny adapter
has not demonstrated a meaningful style benefit and remains optional.

## Supplied results

All new artifacts are under `outputs/quality_tests/`. `before_page_001.png` and
`after_final_page_001.png` use identical raw words; `comparison_final.png` shows
them with a genuine scan reference. `after_review_page_001.png` and
`comparison_review.png` incorporate provisional visual choices for a few words.
Those choices remain pending writer approval, and the three acceleration
occurrences are explicitly marked problematic. `review_session.json` resumes the
prepared 45-occurrence/42-bundle review; an additional fourth candidate for index
16 demonstrates one-word regeneration. `candidates_01.png` through `_08.png`
show the initial three alternatives. `quality_report.json` records all settings,
candidate seeds, image hashes, findings and limitations.

The experiment used the RTX 4080 with 20 DDIM steps and guidance 3, without OCR.
Seven tests plus real-session Tk startup/navigation passed. All earlier data and
image hashes and the original pretrained weights were verified unchanged. Tone
and body alignment improvements are modest; words still drift between printed
and cursive forms and malformed long words remain. No personal-writer similarity
improvement has been established.
