"""Audit a completed Step 7 run without training or running generation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
run = root / 'outputs/finetune/erdem_pilot'
cfg = json.loads((root / 'training/config.json').read_text())
saved_cfg = json.loads((run / 'config.json').read_text())
assert cfg == saved_cfg, 'Run configuration differs from requested configuration'
assert cfg['trainable_prefixes'] == ['style_lin.']
assert cfg['max_steps'] <= 100
rows = list(csv.DictReader((run / 'losses.csv').open(newline='')))
assert [int(r['step']) for r in rows] == list(range(1, len(rows)+1))
assert all(math.isfinite(float(r[k])) for r in rows for k in ['train_loss','gradient_norm'])
assert all(float(r['gradient_norm']) > 0 for r in rows)
vals = [(int(r['step']), float(r['val_loss'])) for r in rows if r['val_loss']]
assert all(math.isfinite(v) for _,v in vals)
initial = json.loads((run/'initial_validation.json').read_text())['loss']
best, bad = initial, 0
for step, val in vals:
    if val < best-cfg['min_delta']: best, bad = val, 0
    else: bad += 1
assert len(rows)<=cfg['max_steps']
assert len(rows)==cfg['max_steps'] or bad>=cfg['patience'], 'Unexpected early termination'
results = {}
loaded = {}
for filename in ['best_adapter.pt','last_adapter.pt']:
    checkpoint = torch.load(run/filename, map_location='cpu', weights_only=True)
    assert checkpoint['config']==cfg
    assert set(checkpoint['adapter'])=={'style_lin.weight','style_lin.bias'}
    assert all(torch.isfinite(t).all() for t in checkpoint['adapter'].values())
    # Match the actual official UNet projection architecture and load strictly.
    projection = torch.nn.Linear(1280,1280)
    projection.load_state_dict({k.removeprefix('style_lin.'):v for k,v in checkpoint['adapter'].items()},strict=True)
    optimizer_steps = [int(v['step']) for v in checkpoint['optimizer']['state'].values()]
    assert not optimizer_steps or all(s==checkpoint['step'] for s in optimizer_steps)
    loaded[filename]=checkpoint
    results[filename]={'step':checkpoint['step'],'validation_loss':checkpoint['validation_loss'],
                       'strict_projection_load':True,'optimizer_steps':optimizer_steps,
                       'sha256':hashlib.sha256((run/filename).read_bytes()).hexdigest()}
assert loaded['last_adapter.pt']['step']==len(rows)
delta = {k:float((loaded['last_adapter.pt']['adapter'][k]-loaded['best_adapter.pt']['adapter'][k]).abs().max())
         for k in loaded['last_adapter.pt']['adapter']}
assert any(v>0 for v in delta.values()), 'Last checkpoint has no learned changes'
protected = json.loads((root/'training/reports/original_sha256.json').read_text())
protected['outputs/erdem_test.png']='39fdb9b7c25435f9da31868de95869deca00392eed044a298792c9cf3db0a012'
for name,expected in protected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
summary=json.loads((run/'summary.json').read_text())
assert summary['steps']==len(rows) and summary['best_validation']==best
# Duration was not logged by the original trainer. File timestamps provide
# an estimate from configuration creation to final summary, excluding loading.
duration=(run/'summary.json').stat().st_mtime-(run/'config.json').stat().st_mtime
report={'existing_completed_run_verified':True,'new_optimizer_steps_this_turn':0,
        'completed_optimizer_steps':len(rows),'early_stopped':bad>=cfg['patience'],
        'initial_validation_loss':initial,'best_validation_by_min_delta':best,
        'lowest_observed_validation_loss':min([initial]+[v for _,v in vals]),
        'validation_history':vals,'checkpoints':results,'trained_checkpoint':'outputs/finetune/erdem_pilot/last_adapter.pt',
        'max_adapter_change_from_saved_best':delta,
        'train_loss_range':[min(float(r['train_loss']) for r in rows),max(float(r['train_loss']) for r in rows)],
        'gradient_norm_range_before_clip':[min(float(r['gradient_norm']) for r in rows),max(float(r['gradient_norm']) for r in rows)],
        'gradient_clip':cfg['gradient_clip'],'all_logged_losses_and_gradients_finite':True,
        'peak_allocated_gib':summary['peak_allocated_gib'],
        'estimated_training_phase_seconds_from_file_timestamps':duration,
        'duration_limitation':'Exact wall duration was not logged. Timestamp estimate excludes model initialization and may be affected by filesystem sync.',
        'protected_source_files_and_baseline_unchanged':True,
        'quality_warning':'Validation changes did not meet min_delta. No generated handwriting quality evaluation was performed.'}
(run/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
