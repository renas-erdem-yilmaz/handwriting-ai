"""Pretrained DiffusionPen adapter training; dry-run is the safe default."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='training/config.json')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--train',action='store_true',help='Explicitly start the bounded Step 7 experiment')
    args = parser.parse_args()
    if args.train and args.dry_run:
        parser.error('Choose either --train or --dry-run')
    cfg = json.loads((ROOT/args.config).read_text())
    os.environ['HF_HOME'] = str(ROOT/cfg['hf_cache'])
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.checkpoint import checkpoint as modern_checkpoint
    from transformers import CanineModel, CanineTokenizer
    from diffusers import AutoencoderKL, DDIMScheduler
    from huggingface_hub import hf_hub_download
    from dataset import WordDataset

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this tested configuration; use a Colab GPU if unavailable')
    device='cuda'
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    vocab=json.loads((ROOT/'diffusionpen-training/letter2index.json').read_text())
    datasets={s:WordDataset(ROOT/cfg['dataset'],s,vocab,cfg['seed']) for s in ('train','val','test')}
    # Exercise every reviewed item, including untouched test images, for data
    # integrity. Test samples are not passed through training/validation loss.
    for split,data in datasets.items():
        for i in range(len(data)):
            item=data[i]
            assert item['image'].shape==(3,64,256)
            assert item['style_images'].shape==(5,3,64,256)
            assert torch.isfinite(item['image']).all() and torch.isfinite(item['style_images']).all()
    print('Dataset sizes:',{s:len(d) for s,d in datasets.items()},flush=True)
    # Snapshot original protected files before even initializing the model.
    protected=json.loads((ROOT/'training/reports/original_sha256.json').read_text())
    protected['outputs/erdem_test.png']='39fdb9b7c25435f9da31868de95869deca00392eed044a298792c9cf3db0a012'
    def verify_originals():
        for name,expected in protected.items():
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=expected:
                raise RuntimeError(f'Protected original changed: {name}')
    verify_originals()

    # Import the official architecture, not Reforge's inference pipeline.
    spec=importlib.util.spec_from_file_location('official_diffusionpen_unet',ROOT/'diffusionpen-training/unet.py')
    official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    # Official custom autograd checkpoint assumes all parameters require grad.
    # PyTorch non-reentrant checkpoint supports frozen parameters correctly.
    official.checkpoint=lambda fn,inputs,params,flag: modern_checkpoint(fn,*inputs,use_reentrant=False) if flag else fn(*inputs)
    spec=importlib.util.spec_from_file_location('official_style_encoder',ROOT/'diffusionpen-training/feature_extractor.py')
    feature=importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    print('Loading pinned pretrained components...',flush=True)
    download=lambda filename:hf_hub_download(cfg['checkpoint_repo'],filename,revision=cfg['checkpoint_revision'],local_files_only=True)
    unet_path=download(cfg['unet_checkpoint']); style_path=download(cfg['style_checkpoint'])
    text=CanineModel.from_pretrained(cfg['text_model'],revision=cfg['text_revision'],local_files_only=True)
    model=official.UNetModel(image_size=(64,256),in_channels=4,model_channels=320,out_channels=4,
                            num_res_blocks=1,attention_resolutions=(1,1),channel_mult=(1,1),num_heads=4,
                            num_classes=339,context_dim=320,vocab_size=79,text_encoder=text,
                            args=SimpleNamespace(interpolation=False,mix_rate=0.0))
    def clean(state):
        result={}
        for k,v in state.items():
            while '.module.' in k: k=k.replace('.module.','.')
            if k.startswith('module.'): k=k[len('module.'):]
            result[k]=v
        return result
    state=clean(torch.load(unet_path,map_location='cpu',weights_only=True))
    # STRICT: mismatches fail; no silently uninitialized weights.
    model.load_state_dict(state,strict=True); del state
    encoder=feature.ImageEncoder(model_name='mobilenetv2_100',pretrained=False,trainable=False)
    state=clean(torch.load(style_path,map_location='cpu',weights_only=True))
    # The released triplet file also stores its training-time writer classifier.
    # It is absent from the feature-only ImageEncoder and not used for inference.
    ignored_style_keys=[k for k in ('classifier.weight','classifier.bias') if k in state]
    for k in ignored_style_keys: del state[k]
    encoder.load_state_dict(state,strict=True); del state
    vae=AutoencoderKL.from_pretrained(cfg['vae_model'],subfolder='vae',revision=cfg['vae_revision'],local_files_only=True)
    tokenizer=CanineTokenizer.from_pretrained(cfg['text_model'],revision=cfg['text_revision'],local_files_only=True)
    model.requires_grad_(False)
    for name,param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in cfg['trainable_prefixes']))
    encoder.requires_grad_(False).eval(); vae.requires_grad_(False).eval()
    # Eval mode keeps every frozen dropout/batchnorm deterministic; gradients
    # still propagate through the frozen UNet to its trainable style projection.
    model.eval(); model.to(device); encoder.to(device); vae.to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError('No trainable parameters')
    total=sum(p.numel() for p in model.parameters()); trained=sum(p.numel() for p in params)
    print(f'Strict checkpoints loaded. Trainable: {trained:,}/{total:,}',flush=True)
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=DDIMScheduler(**cfg['noise_scheduler'])
    loader=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(cfg['seed']))

    def loss_for(batch,gen=None):
        images=batch['image'].to(device)
        with torch.no_grad():
            latents=vae.encode(images).latent_dist.sample(generator=gen)*0.18215
            refs=batch['style_images'].to(device).reshape(-1,3,64,256)
            styles=encoder(refs)
        noise=torch.randn(latents.shape,device=device,generator=gen)
        timesteps=torch.randint(0,1000,(images.shape[0],),device=device,generator=gen).long()
        noisy=scheduler.add_noise(latents,noise,timesteps)
        context=tokenizer(list(batch['text']),padding='max_length',truncation=False,
                          max_length=cfg['max_text_length'],return_tensors='pt').to(device)
        predicted=model(noisy,timesteps=timesteps,context=context,y=None,style_extractor=styles)
        return F.mse_loss(predicted.float(),noise.float()),latents.shape,predicted.shape

    def validation():
        # Same latents/noise/timesteps on every validation pass, separate RNG.
        gen=torch.Generator(device=device).manual_seed(cfg['seed']+10000)
        losses=[]
        with torch.no_grad():
            for batch in DataLoader(datasets['val'],batch_size=1,num_workers=0):
                loss,_,_=loss_for(batch,gen)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite validation loss')
                losses.append(loss.item())
        return sum(losses)/len(losses)

    start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    if not args.train:
        batch=next(iter(loader)); optimizer.zero_grad(set_to_none=True)
        loss,latent_shape,pred_shape=loss_for(batch)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite forward loss')
        loss.backward()
        grads=[p.grad for p in params if p.grad is not None]
        if len(grads)!=len(params) or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing or nonfinite gradients')
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']).item()
        if norm<=0: raise RuntimeError('Zero gradient: adapter cannot learn')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('A frozen parameter received gradients')
        optimizer.zero_grad(set_to_none=True)  # NO optimizer step in Step 6.
        val_loss=validation(); verify_originals()
        # Test adapter checkpoint serialization without updating any weights.
        dry_checkpoint=ROOT/'training/reports/dry_adapter.pt'
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        torch.save({'adapter':adapter,'config':cfg,'optimizer':optimizer.state_dict(),'step':0},dry_checkpoint)
        roundtrip=torch.load(dry_checkpoint,map_location='cpu',weights_only=True)
        assert set(roundtrip['adapter'])==set(adapter)
        assert all(torch.equal(value,roundtrip['adapter'][name]) for name,value in adapter.items())
        report={'success':True,'optimizer_steps':0,'train_loss':loss.item(),'validation_loss':val_loss,
                'gradient_norm_before_clip':norm,'latent_shape':list(latent_shape),'prediction_shape':list(pred_shape),
                'trainable_parameters':trained,'total_parameters':total,'gpu':torch.cuda.get_device_name(0),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-start,
                'strict_unet_load':True,'strict_style_encoder_load':True,'protected_files_unchanged':True,
                'unused_style_classifier_keys':ignored_style_keys,
                'adapter_checkpoint_roundtrip':True,'all_split_items_loaded':True,
                'dataset_sizes':{s:len(d) for s,d in datasets.items()},'training_batch_text':list(batch['text']),
                'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad]}
        write_json(ROOT/'training/reports/dry_run.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return

    # Step 7 only: bounded, separately saved adapter and optimizer state.
    run=ROOT/cfg['run_dir']
    if run.exists(): raise FileExistsError(f'Run directory already exists; choose a fresh run_dir: {run}')
    run.mkdir(parents=True); write_json(run/'config.json',cfg)
    best=validation(); bad=0; write_json(run/'initial_validation.json',{'loss':best})
    def save_checkpoint(name,step,val):
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        torch.save({'adapter':adapter,'optimizer':optimizer.state_dict(),'step':step,'validation_loss':val,
                    'config':cfg,'pretrained_unet':unet_path,'rng_cpu':torch.get_rng_state(),
                    'rng_cuda':torch.cuda.get_rng_state()},run/name)
    save_checkpoint('best_adapter.pt',0,best)
    with (run/'losses.csv').open('w',newline='',encoding='utf-8') as f:
        log=csv.writer(f); log.writerow(['step','train_loss','val_loss','gradient_norm'])
        iterator=iter(loader)
        for step in range(1,cfg['max_steps']+1):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
            optimizer.zero_grad(set_to_none=True); loss,_,_=loss_for(batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
            optimizer.step(); val=None
            if step%cfg['validate_every']==0:
                val=validation(); save_checkpoint('last_adapter.pt',step,val)
                if val < best-cfg['min_delta']:
                    best=val; bad=0; save_checkpoint('best_adapter.pt',step,val)
                else: bad+=1
            log.writerow([step,loss.item(),val,float(norm)]); f.flush()
            print(f'Step {step}: train={loss.item():.6f}, validation={val}',flush=True)
            if bad>=cfg['patience']: print('Early stopping',flush=True); break
        if step%cfg['validate_every']:
            val=validation(); save_checkpoint('last_adapter.pt',step,val)
            if val<best-cfg['min_delta']: save_checkpoint('best_adapter.pt',step,val)
    verify_originals()
    write_json(run/'summary.json',{'steps':step,'best_validation':best,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})

if __name__=='__main__':
    main()
