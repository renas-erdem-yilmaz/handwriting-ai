# Consistent writing size

The default renderer now scales **lowercase x-height**, rather than total ink-box
height. Ascenders and descenders therefore retain their natural extra extent.
Explicit `processing.mode = "legacy"` still reproduces the old algorithm.

In document JSON, configure:

```json
"layout": {
  "x_height_mm": 3.1,
  "size_variation": 0.05,
  "vertical_variation_mm": 0.12,
  "heading_size_scale": 1.25,
  "line_spacing_mm": 8,
  "word_spacing_mm": 2.4
}
```

`x_height_mm` is the physical height of the lowercase body, not the full height of
letters such as b/h/g/y. Set `heading_size_scale` to 1 for identical heading/body
scale. Size variation is bounded at ±5% (pixel rounding can add a fraction of a
pixel). Without an explicit x-height, existing `word_height_mm` is converted to
a nominal x-height using factor .73, preserving a usable old document default.
Keep enough line spacing for full ascender/descender extents; the compositor also
increases line advance when required to prevent overlap.

Foreground measurements composite transparency over white, adapt the ink
threshold, and exclude remote/small noise components. Dot and punctuation
components near the word are retained. This filtering changes measurement masks,
not stroke geometry: composition retains antialiased edges and no thinning,
erosion, dilation or sharpening is applied.

Top/bottom modes across foreground columns estimate body top and baseline.
Ambiguous words (one-letter words, capitals, insufficient support, implausible
extent ratios) fall back to lexical ascender/descender ratios. When at least three
reliable body words of that class exist, the page's median calibrates the fallback.
Heading samples do not influence body calibration. The image is resized once,
uniformly in both directions, preserving stroke proportions and aspect ratio.
Baseline offsets come from the same measured ink, not the original padded canvas.

The previous broad 70–118% whole-box clamp did not enforce a common actual x-height.
The compositor also used to shrink wide words independently. Adaptive mode now
refuses an over-wide word rather than silently changing its writing scale; enlarge
the page or reduce the shared x-height. Legacy mode retains its prior behavior.

The manifest records each word's actual ink bounds, confidence/fallback, estimated
raw and rendered x-height, scale, baseline and page calibration. These are geometric
estimates, not quality or writer-similarity scores. Malformed glyphs and radically
different printed/cursive shapes can still defeat estimation; use candidate review.

New previews and measurements are in `outputs/size_tests/`. They replay the exact
raw word images from `outputs/quality_tests/after_final.json`, without inference or
changes to approved selections. Earlier previews, datasets and checkpoints are
preserved.

```powershell
# Render current persistent candidate selections with the improved default.
.\.reforge\Scripts\python.exe -m page_engine.quality render --session outputs/quality_tests/review_session.json --output outputs/size_tests/my_page

# Geometry fixtures and previous functionality checks.
.\.reforge\Scripts\python.exe -m page_engine.scale_tests
.\.reforge\Scripts\python.exe -m page_engine.tests
.\.reforge\Scripts\python.exe -m page_engine.quality_tests
```

For custom writing size, set the layout settings in your document before creating
a new quality review session. Editing an existing session's document layout only
also affects subsequent rendering; candidate text/seed/model identities must remain
unchanged. Original labels and datasets do not need to be modified.
