# Streamlit v1.0 completion report

Created `app.py`, `app_components/` (editor, settings, preview, export, review,
serialized backend, cached resource, tests and real smoke test),
`.streamlit/config.toml`, `requirements-app.txt`, and root `README.md`.
Extended `handwriting_app/service.py` with notebook size, configurable shared style
profile, ink opacity, bounded variation, scale-aware math font size and reusable
lazy pipeline access. Existing Tkinter/CLI defaults and checkpoint loaders remain.
No model, dataset, annotation or completed experiment was changed.

The browser interface provides editable notes/headings/math, five-word reference
selection with previews, optional existing pilot adapter, page/background/size/
spacing/margins/ink/variation controls, progress/cancellation, all-page previews,
persistent word approval and individual regeneration, and PNG/PDF downloads.
Generation happens only on explicit requests. One cached backend holds one lazy
model instance and serializes operations across browser sessions. Page refresh
recomposes cached words without inferring unrelated words; changed widths can
reflow following lines/pages. Both exports preserve preview pixels, physical
dimensions/page order, and PNG DPI metadata.

## Verified evidence

- 32 automated tests passed: six new Streamlit/backend tests and 26 existing
  desktop/cache/normalization/layout/math regressions. Final combined execution
  captures Tk error dialogs instead of allowing invisible test dialogs to block;
  none occurred. Fixture generation and approvals remain isolated in temporary
  directories and never enter the user's cache or symbol library.
- Localhost server and `/_stcore/health` returned HTTP 200. Browser automation
  could not initialize its sandbox helper. Actual Streamlit AppTest interactions
  were verified, but a live browser screenshot audit was not completed.
- Fresh pretrained CUDA inference and an additional candidate for `Physics` were
  generated through actual Streamlit widget actions. No generation was mocked.
  A pending selection then refreshed the page with no new inference or human
  approval. Existing 32-word samples were reused for A4 and two-page notebook PDFs.
- Pretrained end-to-end workflow took 140.55 seconds, including UI polling,
  loading/regeneration/export; peak allocated GPU memory was 1.033 GiB. The same
  model instance remained active across jobs. Mere reruns/enlargement preserved
  the generated PDF without additional inference.
- Optional 30-update adapter also generated a fresh word through Streamlit on
  CUDA. Its SHA-256 stayed unchanged; no quality improvement is claimed.
- PNG decoded pixels equal the original saved previews, with approximately
  149.987 DPI for the requested 150 DPI. PDF embedded pixels equal those previews,
  correct order/page counts were checked, and PDF pages were rendered and visually
  reviewed. No extra clipping or overlapping content was observed. Inherited
  model letter distortions are visible in the unapproved real-word candidate.
- 1,131 recorded original files/previous review decisions stayed unchanged during
  the real workflow. Dependencies passed `pip check`. Streamlit 1.65.0 and its
  previously absent dependencies were installed in `.reforge`; existing inference
  packages and `.venv` were not upgraded.

Reports/samples live under `outputs/app_tests/streamlit_v1/`:
`automated_tests.json`, `real_integration.json`, `adapter_integration.json`,
`a4/document.pdf`, `notebook/document.pdf`, `real_word/document.pdf`, and page PNGs.
The notebook sample intentionally uses 18 mm line pitch to demonstrate pagination.

## Known limitations / next work

No mathematical symbols are user-approved yet, so examples use explicit font
fallback. The existing assembly tests prove the code path with synthetic fixtures;
they do not prove personalized mathematical resemblance. Collect and approve
actual symbols later; this does not block normal generation/export.

DiffusionPen can misspell/distort letters or vary between printed/cursive forms;
structural flags and candidates cannot guarantee correctness. All real automated
test selections remain pending. The pilot adapter has no established improvement.
PDFs are raster documents. Full restart/disconnection requires reopening a saved
run through existing utilities; session state guarantees reruns, not browser
restart recovery. Cancellation cannot interrupt an active CUDA call. Uncached
long documents can take hours; safe serial inference was kept without unverified
batching. No public deployment or paid services were introduced.

Recommended post-v1 work: saved-document/job recovery UI, more detailed preview
inspection, optional OCR assistance, a larger reviewed symbol/word library, and
measured performance improvements.

Launch from the project root:

```powershell
.\.reforge\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```
