# Personalized mathematical symbol assembly (Step 9.6)

Run from the project root in PowerShell:

```powershell
.\.reforge\Scripts\python.exe -m page_engine.math_symbols collect
.\.reforge\Scripts\python.exe -m page_engine.math_symbols render --expression "v_0^2 + 2ax" --output outputs/math_tests/my_equation.png
.\.reforge\Scripts\python.exe -m page_engine.math_symbols examples --output outputs/math_tests/after_collection
.\.reforge\Scripts\python.exe -m page_engine.math_symbols.tests
```

Use a fresh output path for each run. Existing PNGs and manifests are preserved.
`--library path` is a global option, **before** the subcommand, for separate libraries.
The default library is `.reforge-cache/math_symbols/`, already excluded from Git.
No dependencies were added. Tkinter, Pillow, NumPy and the existing Matplotlib
font fallback are local; collecting/rendering math does not initialize DiffusionPen.

## Collect and explicitly approve

Open a scan (the first `data/pages/` image is initially displayed). Zoom to 100%
or 200%, scroll to a clean symbol, and drag a tight rectangular selection.
Enter **its exact label yourself**. Nothing infers a label from the image.
Multiple independently reviewed crops may share a label. Latin variables and
digits use their literal character; Greek letters use literal Unicode. Use `−`
for subtraction, `×` for multiplication, `⇒` for implication, and `fraction_bar`
for a straight fraction stroke. `-` is also collectable but input subtraction
is normalized to the mathematically equivalent minus symbol `−`.

The preview shows the untouched crop above its grayscale ink-only copy. Adjust
the ink cutoff (default 220) to remove pale paper/grid, while checking that faint
strokes survive. No thinning, sharpening, morphological cleanup, or letter-shape
inference is used. Dark grid lines and overlapping handwriting cannot safely be
removed automatically; choose a cleaner occurrence or crop more tightly.

The red line marks the baseline as a fraction of the **tight processed ink height**.
Set `Height / prose x-height` to ~1 for body-height lowercase letters, ~1.4 for
capitals/digits/ascenders, ~1.4 for descenders with baseline ~0.7, and ~0.1 for
fraction bars. These are starting suggestions, not inferred measurements.
Operators should have baselines set to align their center above the writing
baseline; review preview equations and adjust. Greek shapes vary: inspect each.
Scripts use the same approved glyphs, resized to 64%; no separate font is needed.

Save drafts freely. To enable rendering, check “I checked BOTH the symbol label
and crop” and press **Approve checked crop and label**. Editing/saving a draft
revokes its approval. Existing variants can be selected and reviewed again; the
raw PNG is never overwritten. Library metadata retains source path/hash, original
pixel rectangle, raw crop hash, processing settings, approval time and edit history.
Approved-image hash mismatches raise an error rather than silently use changed ink.
Original pages and datasets are opened read-only. Closing/reopening resumes saved
variants; unsaved rectangle selections are not persisted.

## Restricted syntax and mathematical meaning

Supported: digits, decimal points, single-letter Latin/Greek variables, `+`, `-`
or `−`, `*` or `×`, `÷`, `=`, `⇒`, `→`, `≤`, `≥`, `≠`, `±`, parentheses/brackets,
slash fractions, `^` superscripts and `_` subscripts. Whitespace is insignificant.
Adjacent letters mean implicit multiplication, **not** named functions:
`ma` means `m × a`. This is not a general LaTeX or symbolic-algebra parser.
No automatic algebraic rewriting occurs.

Examples: `F = ma`, `a = F/m`, `v = v_0 + at`, `x = 1/2 at^2`,
`θ = α + β`, `v^2 = v_0^2 + 2ax`, `(a+b)/(c-d)`, `v_0^{2+1}`.
Unbraced scripts consume one character: use `x^{12}`, not `x^12` for power twelve.
Multiplication and division (including slash) are evaluated left to right;
slash places the accumulated left product over the following factor:
`1/2 at^2` is `(1/2) × a × t²`; `F/ma` is `(F/m) × a`.
**Use `F/(ma)` when the whole product belongs in the denominator.**
Visible parentheses/brackets are retained; ordinary brace groups become visible
parentheses to preserve meaning. Braces inside script notation delimit the script
without drawing braces. Unary signs, nested fractions and combined scripts work.
Malformed supported syntax raises an error; expressions outside this grammar
(including existing `\frac`, `\sqrt`, etc.) retain the original Mathtext path.
Unsupported original Mathtext commands still raise an error.

Approved glyphs scale uniformly with aspect ratios preserved; independent variants
are selected reproducibly by seed with bounded ±2.5% size variation (maximum 5%).
Glyph baselines come from reviewed metadata. Scripts and fractions expand their
canvas to avoid clipping. Fraction bars alone extend horizontally, preserving their
thickness; this is an assembled crop, not a newly generated continuous pen stroke.
Long fractions may benefit from separate longer bar variants. Parentheses are
ordinary reviewed glyphs, not automatically stretched tall delimiters.

**Every required symbol must have an approved variant.** Otherwise the entire
equation uses the original STIX font renderer. The parsed equation is serialized
with explicit fractions/scripts so its meaning survives fallback. This avoids a
patchwork of handwriting and font glyphs. No visually similar replacement is made:
`θ`, `φ`, `0`, `O`, `×`, `x`, `⇒` and `→` have distinct inventory labels.
JSON records source, input, canonical expression, missing symbols, seed and library
hash. Handwritten output additionally records each crop ID/hash and placement.

## Existing page layout integration

Math blocks and `$...$` inline equations use the new engine by default. Add to a
document JSON (all fields optional):

```json
"math": {
  "library": ".reforge-cache/math_symbols",
  "mode": "auto",
  "seed": 9600,
  "variation": 0.025,
  "size_scale": 1.0
}
```

`mode: "font"` explicitly restores the prior renderer. Handwritten math uses the
page's configured `layout.x_height_mm` times `size_scale`, aligning with prose.
The existing math font point setting controls fallback. Page footers identify
font fallback where present, and placement manifests record complete provenance.
Word generation, caches, review and optional adapter selection are unchanged.
The examples command replays the first two physics paragraphs and title from the
existing quality preview's genuine cached generations, preserving seeds/model
identity; it performs no new inference and does not approve any prose or symbol.

## Current readiness and limitations

No symbol is usable until **you** review and approve it. Initial example equations
are deliberately labeled font fallback if the library is empty. Their generation
does not prove personal mathematical resemblance. After collecting the missing
inventory shown in `outputs/math_tests/report.json`, rerun examples to a fresh
directory. Unit tests exercise assembly using explicitly synthetic temporary
fixtures; those fixtures are never installed in the user's symbol library.

Per-glyph composition lacks the ligatures and pen flow of a continuous handwritten
equation. Visual scale/baselines depend on your reviewed settings and clean crops.
The existing DiffusionPen prose can still contain misspelled/malformed letters;
reviewing mathematics does not fix that model limitation. No model retraining or
final application is included.
