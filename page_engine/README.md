# Step 9: local word generation and page previews

For adaptive consistency, three-candidate generation and the local Tkinter word
reviewer, see [QUALITY.md](QUALITY.md). The original commands below remain available.
For current x-height-based writing-size controls, see [SCALE.md](SCALE.md).

Run from the main project directory in PowerShell. All original data, training
weights, Reforge sources, baseline and evaluation images remain read-only.

```powershell
.\.reforge\Scripts\python.exe -m page_engine analyze
.\.reforge\Scripts\python.exe -m page_engine render --document page_engine/examples/newton.json --output outputs/layout_tests/my_newton
.\.reforge\Scripts\python.exe -m page_engine render --document page_engine/examples/physics.json --output outputs/layout_tests/my_physics
.\.reforge\Scripts\python.exe -m page_engine render --document page_engine/examples/notes.json --output outputs/layout_tests/my_notes
```

Choose a fresh output prefix each time; existing previews are preserved. Re-running
the same document/settings with a new prefix uses cached word images without loading
GPU models. Default: original pinned pretrained DiffusionPen, 20 DDIM steps,
guidance 3, one candidate per attempt, at most one structural retry, no OCR.
Optional `--adapter outputs/finetune/erdem_pilot/last_adapter.pt` uses the strictly
checked style projection. No training occurs. Model/reference hashes and sampling
settings namespace the cache. Changing a word occurrence seed gives a distinct
cached variation. RNGs reset per attempt; exact cross-hardware reproduction is not promised.

## Manual review and regeneration

```powershell
.\.reforge\Scripts\python.exe -m page_engine contact-sheet --manifest outputs/layout_tests/my_newton.json --output outputs/layout_tests/review.png
.\.reforge\Scripts\python.exe -m page_engine review CACHE_ID approved
.\.reforge\Scripts\python.exe -m page_engine review CACHE_ID rejected
.\.reforge\Scripts\python.exe -m page_engine regenerate --manifest outputs/layout_tests/my_newton.json --word-index 0 --seed 123456 --output outputs/layout_tests/my_newton_revised
```

Find occurrence indexes and cache IDs in the manifest/contact sheet. Regeneration
preserves the previous candidate and preview and reuses all unchanged words. Review
status is shared only by that exact cached candidate. `--require-approved` refuses
unapproved words. These are previews: structural screening cannot detect every
malformed word or establish spelling. Retries never guarantee correct spelling.
For an adapter preview, supply `--adapter` to `render` and `--overrides seeds.json`
(e.g. `{"0": 123456}`) rather than the pretrained-only regeneration shortcut.

`--ocr` optionally invokes the existing local TrOCR validator. It may require its
cached weights; it is never loaded by default. OCR mismatch triggers a bounded
retry, not a claim of correctness. OCR is a lexical check, not a writer similarity
metric. Reforge's implicit OCR is disabled in a restored, scoped override; this
pipeline is single-threaded and must not be shared concurrently.

## Documents and layout

JSON blocks have `kind`: `heading`, `paragraph`, or `math`, and `text`. Blank
paragraph separation is expressed as separate paragraph blocks. Inline math uses
`$...$`; display math blocks omit dollars. Plain prose accepts Latin words with
apostrophes/hyphens and terminal punctuation. Unsupported tokens fail explicitly:
put numbers, Greek letters and operators in math runs rather than corrupting them
through DiffusionPen. Examples include `$α$`, `$v_0$`, `$t^2$`, and
`$\\frac{F}{m} ⇒ a$`. Unicode mappings cover the requested Greek letters, arrows,
several common operators and basic superscript/subscript digits. General TeX syntax
must be supported by Matplotlib Mathtext; unsupported syntax raises an error.
No symbolic algebra or correctness proof is performed.

Math now assembles explicitly approved handwritten symbol crops when the entire
equation's inventory is available; otherwise it uses STIX standard-font fallback.
See [math_symbols/README.md](math_symbols/README.md) for the collection tool,
restricted syntax, approval requirements and commands. Every page has an explicit
footer and every math placement has provenance.
Reforge also synthesizes trailing punctuation through its existing font path.
Unsupported equations are never silently dropped or replaced with bogus output.

`layout` settings: `page` (`a4` or `notebook`, default B5 notebook size), optional
`size_mm`, `dpi`, `margins_mm` [left,right,top,bottom], `line_spacing_mm`,
`word_height_mm`, `word_spacing_mm`, `indent_mm`, `paragraph_extra_lines`,
`heading_align` (`center`/`left`), and `math_font_points`. Normalization uses actual
ink bounds and bounded 3% size/1-pixel alignment variation. Short words without
ascenders have smaller ink height; aspect ratio is preserved. Wrapping uses actual
normalized widths. Pale background texture is removed from composition copies;
raw cached candidates are preserved. Equations stay indivisible; excessively wide equations fail
rather than becoming unreadably small. Tall fractions increase line advance.

The reusable `profiles/scanned.json` retains measurements and source hashes for
21 scans. Grid removal, dark ink thresholds, margins, heading position and line
bands are CV heuristics, affected by diagrams/equations. Conservative template
clamps are documented in the profile. This is not automatic reconstruction of
the original handwritten page structure.

Dependencies: existing `.reforge` stack plus `matplotlib==3.11.2`. All inference is
offline using existing checkpoint cache; GPU selected when available. Cache and
outputs are covered by existing `.gitignore`. No final application is included.

## Verified outputs

`outputs/layout_tests/newton_final_page_001.png`, `physics_final_page_001.png`,
and `notes_final_page_001.png` are actual pretrained generations, normalized and
composed from the raw word cache. `multipage_page_001.png` through `_003.png`
demonstrate 450-word pagination using replayed generated candidates. The source
manifest explicitly identifies this as a layout stress test.

`newton_review.png` is a word contact sheet; `newton_cli_regenerated_page_001.png`
demonstrates the documented regeneration command. New candidates remain pending
human review. The approval persistence test restores pending status and is not a
human approval. `test_report.json` records hashes, configuration, measured GPU
allocation, tests and visual limitations. Run unit tests with:

```powershell
.\.reforge\Scripts\python.exe -m page_engine.tests
```

The one-time `run_tests` and `finalize` modules preserve existing completed outputs;
use the normal `render` CLI with a fresh prefix for subsequent previews.
