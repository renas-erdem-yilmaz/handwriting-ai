# Step 6: pretrained DiffusionPen fine-tuning pilot

Step 6 passed a real GPU forward/backward batch and four validation targets.
No optimizer step was taken and no baseline weights were changed.

## Source implementation and checkpoint compatibility

The separate `diffusionpen-training/` checkout is the official
https://github.com/koninik/DiffusionPen repository at commit
`059e2f2bcb7372231038719e955597917c1a8783`.
Its `train.py` uses epsilon MSE on SD 1.5 VAE latents scaled by 0.18215,
CANINE-C text conditioning and five style word images per target. Its original
launcher assumes IAM cached data, DataParallel, wandb and a multiwriter setup.
We use its actual `unet.py` and `feature_extractor.py` with a separate CSV loader
and training entry point; Reforge remains an inference implementation.

The existing cache supplies the official IAM EMA checkpoint and pretrained
triplet style encoder at Hugging Face `konnik/DiffusionPen`, revision
`5636a68365001435ec76553114313c22c6f1ca07`. DataParallel key prefixes are
normalized, then every UNet weight is loaded strictly. Only the style file's
unused `classifier.weight` and `classifier.bias` are explicitly omitted;
every feature-encoder weight loads strictly. The 339-writer embedding is
preserved, frozen and bypassed by pretrained visual style features.

The official custom autograd checkpoint helper assumes all parameters require
grad. Our entry point substitutes PyTorch's non-reentrant helper at runtime so
frozen weights can propagate gradients to the style projection safely. No
upstream or Reforge files were changed for Step 6.

## Data and review policy

The original CSV columns `filename,text,type,page` and their split `images/`
directories are audited without writing to them. The current prepared splits
contain **48 region samples (33 train, 7 val, 8 test)**, rather than 53 usable
entries. No original labels are normalized, transliterated or discarded.

`reports/label_audit.json` includes every original split sample, dimensions,
multiword status, visual line-contamination observations, mathematical notation
and unsupported Unicode characters. `reports/unsupported_samples.csv` and
`.json` list all **31 unsupported samples (23 train, 4 val, 4 test)** with exact
labels and codepoints. Official `letter2index.json` defines the checkpoint's
trained character coverage; CANINE's ability to tokenize Unicode does not
establish that the diffusion checkpoint can draw Greek characters or equations.
Supported slash characters also do not imply support for spatial fractions.

43 original samples have multiple whitespace-separated words. Whole lines
must not be resized into word targets. We instead prepared a deliberately
small, visually reviewed subset with explicit boxes in `reviewed_words.json`.
Only those boxes are copied into `data/finetune_words/`: **18 train, 4 val,
6 test words**. Original misspellings such as `dimention` remain unchanged.
The contact sheets `train_review.png`, `val_review.png`, `test_review.png`
show the actual tensors' image content. Other regions remain pending review;
long words and mathematical/multiline crops are not automatically aligned.
To expand, inspect a parent image, add exact original word text and box to
`reviewed_words.json`, regenerate, then visually check the sheets again.
Keep an entire parent region in its original split.

References come exclusively from reviewed training images, contain at least
four characters and exclude the training target image. Five references are
selected deterministically per target. Validation/test images are never used
as training references. The Step 5 reference originated in the test split;
it is intentionally not reused here. Original splits share pages. The reviewed
pilot has page 1 training and page 2 validation, but page 1 test, so it still
cannot establish broad page/writer generalization.

## Configuration and safeguards

`config.json` trains only pretrained `style_lin.weight` and `style_lin.bias`
(1,639,680 of 169,908,868 UNet parameters). The pretrained style encoder,
CANINE, VAE and remaining UNet layers stay frozen, in eval mode. No component
is trained from random initialization. FP32, batch 1, LR 1e-6, weight decay
0.01, gradient norm cap 1.0, maximum 100 steps. Validation uses fixed noise,
timesteps and VAE sampling with a separate seeded generator every 10 steps.
Early stopping follows three validations without improvement. The initial
pretrained validation result is the baseline: the saved best adapter remains
unchanged if fine-tuning never improves it. This is an exploratory pilot,
not a recommendation to fit the entire model to 18 words.

Step 7 writes `losses.csv`, initial validation, config, optimizer state,
best/last adapters and a summary under `outputs/finetune/erdem_pilot/`.
An existing run directory is refused. Adapter checkpoints contain only the
two changed tensors plus provenance and optimizer metadata, and must be
applied to the same pretrained base; they are not standalone Reforge weights.
Original split CSV/image hashes and baseline PNG hash are checked before and
after execution. Test data loads for integrity checks but is not used for
optimization or early stopping.

## Environment and actual dry-run results

Existing Python 3.11 `.reforge` is reused. **No dependency changes were made
for Step 6.** Torch 2.5.1+cu124, torchvision 0.20.1+cu124, diffusers 0.32.2,
transformers 4.46.3, huggingface-hub 0.36.0, tokenizers 0.20.3.
The official launcher's wandb/skimage dependencies are not needed by this
separate entry point.

Dry run on RTX 4080 Laptop GPU (12 GB): strict pretrained loading passed,
one `output` target produced latent/prediction shape `[1,4,8,32]`, forward
MSE 1.243666, finite nonzero backward gradient norm 3.357805 before clipping.
All frozen weights received no gradients. Four-target validation loss
0.015522. These losses use different noise/timesteps; they are not directly
comparable and do not demonstrate improved handwriting. Optimizer steps: 0.
Peak PyTorch allocated memory 1.026 GiB, reserved 1.105 GiB; CUDA context and
desktop GPU use are additional. Allow 3–4 GB headroom for this narrow batch-1
configuration; full-UNet fine-tuning would require substantially more memory
and has not been tested. See `reports/dry_run.json` for measured details.

## Commands (PowerShell, main project directory)

Rebuild the derived dataset/reports, retaining original labels and images:

```powershell
.\.reforge\Scripts\python.exe .\training\prepare_words.py
.\.reforge\Scripts\python.exe .\training\finetune.py --config training/config.json --dry-run
```

**Step 7 command — not executed during Step 6:**

```powershell
.\.reforge\Scripts\python.exe .\training\finetune.py --config training/config.json --train
```

`--train` is mandatory; without it the entry point defaults to a dry run.
Model loading uses pinned, cached revisions offline. Keep `.reforge-cache/`,
`data/`, `outputs/`, and the upstream checkout ignored by Git. Reforge,
pretrained checkpoints, originals and `outputs/erdem_test.png` remain intact.

Before drawing conclusions, expand reviewed word annotations from more of the
21 pages and create parent/page-disjoint holdouts. A frozen style encoder and
small adapter are technically sound here; training a new style encoder with
one writer is not. Equations require a different model or explicit dedicated
data/model support, rather than silently replacing their symbols.
