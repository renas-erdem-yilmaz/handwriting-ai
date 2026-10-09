"""Paired Step 8 inference and measured comparison; never trains a model."""
import csv
import hashlib
import json
import os
import random
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'training/config.json').read_text())
os.environ['HF_HOME']=str(ROOT/CFG['hf_cache'])
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
sys.path.insert(0,str(ROOT/'reforge'))
import torch
from reforge.model import weights
from reforge.model import encoder as encoder_module
from reforge.model import generator
from reforge.compose import render
from reforge.pipeline import run
from reforge.evaluate.ocr import ocr_read

OUT=ROOT/'outputs/evaluation'
TEXTS=['This is my first generated handwriting test.',
       'Artificial intelligence is changing the world.',
       'The velocity and acceleration change over time.']
ADAPTERS={'baseline':ROOT/'outputs/finetune/erdem_pilot/best_adapter.pt',
          'finetuned':ROOT/'outputs/finetune/erdem_pilot/last_adapter.pt'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def panel(index,text,paths,reference):
    images=[Image.open(paths[k]).convert('RGB') for k in ('baseline','finetuned')]
    width=max(600,max(im.width for im in images)); height=max(im.height for im in images)
    canvas=Image.new('RGB',(width*2+60,height+240),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    draw.text((20,12),f'Comparison {index}: {text}',font=font,fill='black')
    draw.text((20,44),'GENUINE HANDWRITING REFERENCE (dataset crop; not generated)',font=font,fill='black')
    ref=Image.open(reference).convert('RGB');ref.thumbnail((canvas.width-40,95));canvas.paste(ref,(20,72))
    for col,(label,im) in enumerate(zip(['PRETRAINED / BASELINE ADAPTER (step 0)','FINE-TUNED ADAPTER (30 optimizer steps)'],images)):
        x=20+col*(width+20);draw.text((x,184),label,font=font,fill='black');canvas.paste(im,(x,220))
    canvas.save(OUT/f'comparison_{index}.png')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'evaluation_report.json').exists() and not args.resume:raise FileExistsError('Preserve existing evaluation; use --resume for an incomplete run')
    if (OUT/'evaluation_report.json').exists() and args.resume:
        previous=json.loads((OUT/'evaluation_report.json').read_text())
        if previous.get('completed'):raise RuntimeError('Evaluation already completed; preserve the results')
        assert previous['training_config']==CFG
        for label,path in ADAPTERS.items():assert previous['checkpoint_sha256'][label]==digest(path)
    os.chdir(ROOT/'reforge')  # Resolve the same repository punctuation font.
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    train=list(csv.DictReader((ROOT/'data/finetune_words/train/labels.csv').open(encoding='utf-8')))
    chosen=[]
    for word in ['dimention','should','equal','will','return']:
        r=next(r for r in train if r['text']==word)
        chosen.append(ROOT/'data/finetune_words/train/images'/r['filename'])
    reference=ROOT/'data/crops/page_001_region_011_crop_0011.png'
    base_path=Path(weights.download_unet_weights());style_path=Path(weights.download_style_encoder_weights())
    protected=json.loads((ROOT/'training/reports/original_sha256.json').read_text())
    protected['outputs/erdem_test.png']='39fdb9b7c25435f9da31868de95869deca00392eed044a298792c9cf3db0a012'
    extra=[base_path,style_path,*ADAPTERS.values(),reference,*chosen]
    fingerprints={str(p):digest(p) for p in extra}
    print('Loading original pretrained UNet and checking all keys...',flush=True)
    model=weights.load_unet(str(base_path),device='cuda')
    base_state=weights.strip_module_prefix(torch.load(base_path,map_location='cpu',weights_only=True))
    model.load_state_dict(base_state,strict=True)
    names={n for n,p in model.named_parameters() if n.startswith('style_lin.')}
    assert names=={'style_lin.weight','style_lin.bias'}
    assert tuple(model.style_lin.weight.shape)==(1280,1280) and tuple(model.style_lin.bias.shape)==(1280,)
    loaded={}
    for label,path in ADAPTERS.items():
        c=torch.load(path,map_location='cpu',weights_only=True)
        assert c['config']==CFG and set(c['adapter'])==names
        assert c['step']==(0 if label=='baseline' else 30)
        for n,t in c['adapter'].items():
            assert t.shape==base_state[n].shape and t.dtype==base_state[n].dtype and torch.isfinite(t).all()
        if label=='baseline':assert all(torch.equal(t,base_state[n]) for n,t in c['adapter'].items()),'Step-0 adapter differs from pretrained base'
        loaded[label]=c
    del base_state
    # Enforce exact loading of the frozen feature encoder too.
    class StrictEncoder(encoder_module.StyleEncoder):
        def __init__(self,checkpoint_path=None):
            super().__init__(None)
            s=weights.strip_module_prefix(torch.load(checkpoint_path,map_location='cpu',weights_only=True))
            for key in ['classifier.weight','classifier.bias']:s.pop(key,None)
            self.encoder.load_state_dict(s,strict=True)
    encoder_module.StyleEncoder=StrictEncoder
    model.requires_grad_(False).eval()
    weights.load_unet=lambda *a,**k:model
    vae=weights.load_vae(device='cuda');vae.requires_grad_(False)
    tokenizer=weights.load_tokenizer()
    weights.load_vae=lambda *a,**k:vae
    weights.load_tokenizer=lambda:tokenizer
    state={'seed':0,'word_idx':0,'words':[]}
    original_generate=generator.generate_word
    def paired_word(*a,**k):
        seed_all(state['seed']+state['word_idx']);state['word_idx']+=1
        return original_generate(*a,**k)
    generator.generate_word=paired_word
    original_compose=render.compose_words
    def capture_words(images,words,*a,**k):
        state['words']=[(w,im.copy()) for w,im in zip(words,images) if im is not None and w is not None]
        return original_compose(images,words,*a,**k)
    render.compose_words=capture_words
    report={'completed':False,'training_performed':False,'gpu':torch.cuda.get_device_name(0),
            'model':'Official pretrained DiffusionPen IAM EMA, Reforge inference architecture strictly verified',
            'training_config':CFG,'checkpoint_paths':{k:str(p.relative_to(ROOT)) for k,p in ADAPTERS.items()},
            'checkpoint_sha256':{k:fingerprints[str(p)] for k,p in ADAPTERS.items()},
            'adapter_parameters':{'style_lin.weight':[1280,1280],'style_lin.bias':[1280]},
            'baseline_equals_pretrained_projection':True,'strict_checkpoint_compatibility':True,
            'style_references':[str(p.relative_to(ROOT)) for p in chosen],
            'genuine_reference_crop':str(reference.relative_to(ROOT)),
            'settings':{'steps':50,'guidance_scale':3.0,'candidates':3,'page_ratio':'auto','precision':'float32',
                        'composition':'Same unmodified Reforge compose_words implementation and repository font',
                        'seed_policy':'Reset all RNGs for each run and each corresponding word; word seed = sentence seed + word index'},
            'metrics_interpretation':'OCR evaluates lexical readability, not writer identity. Layout scores are CV heuristics. Pixel differences compare generated outputs, not handwriting-style similarity.',
            'limitations':['18 training words and four validation words; single writer.',
                           'Original splits share pages; references are training-only and this is not independent page-level style validation.',
                           'Only three sentences and one paired seed per sentence; adaptive OCR retries can select different candidates.',
                           'Long words are split and stitched by Reforge; trailing punctuation uses its font fallback.',
                           'No calibrated writer-similarity model or independent human rating is available; no writer-similarity score is claimed.',
                           'Training-word overlap and novel outputs cannot by themselves prove or disprove memorization.'],
            'comparisons':[]}
    def convert(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    def save(): (OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,default=convert),encoding='utf-8')
    report['ocr_normalization']='Case-insensitive lexical words; trailing punctuation stripped; CER is summed word edit distance / target character count.'
    for index,text in enumerate(TEXTS,1):
        entry={'sentence':text,'seed':42000+index*100,'variants':{}}
        paths={}
        for label in ['baseline','finetuned']:
            path=OUT/f'{label}_{index}.png'
            if path.exists():
                if not args.resume:raise FileExistsError(path)
                meta=path.with_suffix('.json')
                if meta.exists():
                    entry['variants'][label]=json.loads(meta.read_text())
                    paths[label]=path
                    assert entry['variants'][label]['sha256']==digest(path)
                    continue
                if index!=1:raise RuntimeError(f'Missing recovery metadata: {path}')
                # First pair completed before serialization failed. Keep its
                # PNGs; evaluate explicitly hand-inspected final-image regions.
                boxes=[(110,0,284,75),(285,0,343,75),(343,0,438,75),(27,85,158,166),
                       (161,85,431,166),(24,176,300,260),(300,176,493,260)]
                im=Image.open(path).convert('L')
                reads=[{'target':w,'recognized':ocr_read(np.array(im.crop(box)))} for w,box in zip(text.split(),boxes)]
                norm=lambda s:s.lower().strip().rstrip('.,;:!?')
                entry['variants'][label]={'output':str(path.relative_to(ROOT)),'checkpoint_step':loaded[label]['step'],
                    'size':list(im.size),'sha256':digest(path),'ocr_words':reads,
                    'lexical_character_error_rate':sum(distance(norm(r['target']),norm(r['recognized'])) for r in reads)/sum(len(norm(r['target'])) for r in reads),
                    'lexical_word_accuracy':sum(norm(r['target'])==norm(r['recognized']) for r in reads)/len(reads),
                    'ocr_source':'Hand-inspected final PNG word regions, matched for both variants',
                    'ocr_regions':boxes,'layout_heuristics':{},'generation_reported_seconds':395 if label=='baseline' else 399,
                    'recovery':'PNG preserved after JSON serialization failure; original in-memory layout measurements unavailable'}
                meta.write_text(json.dumps(entry['variants'][label],indent=2,default=convert),encoding='utf-8')
                paths[label]=path;continue
            # Adapter application must cover exactly every selected parameter.
            model.style_lin.load_state_dict({n.removeprefix('style_lin.'):t for n,t in loaded[label]['adapter'].items()},strict=True)
            state.update(seed=entry['seed'],word_idx=0,words=[]);seed_all(entry['seed'])
            start=time.monotonic();print(f'Generating sentence {index}: {label}',flush=True)
            # OpenCV's Windows imread cannot handle Unicode absolute paths.
            # Relative paths from the repository contain only ASCII here.
            result=run(style_image_paths=[os.path.relpath(p,ROOT/'reforge') for p in chosen],text=text,output_path=str(path),num_steps=50,
                       guidance_scale=3.,num_candidates=3,device='cuda',page_ratio='auto')
            reads=[{'target':w,'recognized':ocr_read(im)} for w,im in state['words']]
            norm=lambda s:s.lower().strip().rstrip('.,;:!?')
            errors=sum(distance(norm(r['target']),norm(r['recognized'])) for r in reads)
            chars=sum(len(norm(r['target'])) for r in reads)
            q=result['quality_scores']
            allowed=['baseline_alignment','word_height_ratio','layout_regularity','punctuation_visibility','background_cleanliness','stroke_weight_consistency']
            with Image.open(path) as im: im.verify()
            with Image.open(path) as im: size=list(im.size)
            entry['variants'][label]={'output':str(path.relative_to(ROOT)),'checkpoint_step':loaded[label]['step'],
                                    'seconds':time.monotonic()-start,'size':size,'sha256':digest(path),'ocr_words':reads,
                                    'lexical_character_error_rate':errors/chars,'lexical_word_accuracy':sum(norm(r['target'])==norm(r['recognized']) for r in reads)/len(reads),
                                    'layout_heuristics':{k:q[k] for k in allowed if k in q},'word_positions':result['word_positions']}
            entry['variants'][label]['ocr_source']='Final normalized word images passed to the compositor (before 2x page upscaling)'
            path.with_suffix('.json').write_text(json.dumps(entry['variants'][label],indent=2,default=convert),encoding='utf-8')
            paths[label]=path;save()
        a=np.array(Image.open(paths['baseline']).convert('L'));b=np.array(Image.open(paths['finetuned']).convert('L'))
        h=max(a.shape[0],b.shape[0]);w=max(a.shape[1],b.shape[1]);aa=np.full((h,w),255);bb=aa.copy()
        aa[:a.shape[0],:a.shape[1]]=a;bb[:b.shape[0],:b.shape[1]]=b
        entry['paired_pixel_difference']={'white_padded_mean_absolute_difference':float(np.abs(aa-bb).mean()),
                                          'changed_pixel_fraction':float((aa!=bb).mean()),'exactly_identical':bool(np.array_equal(a,b))}
        entry['training_word_overlap']=[w for w in text.split() if any(norm(w)==norm(r['text']) for r in train)]
        panel(index,text,paths,reference);entry['comparison_image']=f'outputs/evaluation/comparison_{index}.png'
        report['comparisons'].append(entry);save()
    for path,expected in fingerprints.items():assert digest(Path(path))==expected,path
    for path,expected in protected.items():assert digest(ROOT/path)==expected,path
    report['protected_originals_unchanged']=True;report['completed']=True
    report['visual_findings']='Pending visual review of the generated comparison panels.'
    save();print('All six generations and three comparison panels verified. No protected files changed.',flush=True)

if __name__=='__main__':main()
