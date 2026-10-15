"""Validated documents, serial cached inference, cancellation and lossless PDF export."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import io
import math
import re
import threading
import uuid
import time
from PIL import Image
from page_engine.common import ROOT, read_json, write_json as atomic_json, sha
from page_engine.layout import build, tokens
from page_engine.math_symbols.renderer import render as render_math
from page_engine.model import Pipeline
from page_engine.candidates import CandidateStore

PAGE_MM={'A4':(210,297),'Letter':(215.9,279.4),'Notebook':(176,250)}
WORD=re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*[.,;:!?]?")

class Cancelled(Exception):pass

def write_json(path,data):
    for attempt in range(5):
        try:atomic_json(path,data);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05*(attempt+1))

@dataclass(frozen=True)
class Settings:
    page: str='A4'
    background: str='blank'
    margins: tuple=(16,16,22,18)  # left, right, top, bottom in mm
    line_spacing: float=8
    x_height: float=3.1
    word_spacing: float=2.4
    paragraph_gap: float=.8
    seed: int=9500
    steps: int=20
    guidance: float=3.0
    candidates: int=1
    adapter: str|None=None
    device: str='auto'
    dpi: int=150
    style_config: str='page_engine/styles/erdem.json'
    opacity_target: float=.82
    natural_variation: float=.025

    def validate(self):
        if self.page not in PAGE_MM or self.background not in ('blank','lined'):raise ValueError('Choose A4/Letter/Notebook and blank/lined paper')
        numbers=(*self.margins,self.line_spacing,self.x_height,self.word_spacing,self.paragraph_gap,self.guidance,self.opacity_target,self.natural_variation)
        if len(self.margins)!=4 or not all(math.isfinite(float(v)) for v in numbers):raise ValueError('Settings must be finite numbers')
        if not all(6<=v<=70 for v in self.margins):raise ValueError('Each margin must be 6–70 mm')
        if not 1.5<=self.x_height<=6:raise ValueError('Writing x-height must be 1.5–6 mm')
        if not self.x_height/.73+1.1<=self.line_spacing<=25:raise ValueError('Increase line spacing: allow at least x-height / 0.73 + 1.1 mm')
        if not 0<=self.word_spacing<=10 or not 0<=self.paragraph_gap<=4:raise ValueError('Word spacing must be 0–10 mm; paragraph gap 0–4 lines')
        if not .55<=self.opacity_target<=.95 or not 0<=self.natural_variation<=.05:raise ValueError('Ink opacity must be 0.55–0.95; natural size variation 0–0.05')
        if self.device not in ('auto','cuda','cpu') or not 72<=self.dpi<=300:raise ValueError('Unsupported device or DPI')
        if not isinstance(self.seed,int) or not 0<=self.seed<2**32:raise ValueError('Seed must be an integer from 0 to 4294967295')
        if not isinstance(self.steps,int) or not 1<=self.steps<=200 or not 0<self.guidance<=20 or self.candidates not in (1,2,3):raise ValueError('Invalid inference settings')
        width,height=PAGE_MM[self.page];left,right,top,bottom=self.margins
        if width-left-right<50 or height-top-bottom<self.line_spacing*3:raise ValueError('Margins leave too little room for writing')

def parse_document(text,settings):
    settings.validate()
    if not isinstance(text,str) or not text.strip():raise ValueError('Enter some notes before generating')
    if len(text)>100_000:raise ValueError('Please split notes longer than 100,000 characters into separate documents')
    blocks=[]
    for paragraph in re.split(r'\n\s*\n',text.replace('\r\n','\n').replace('\r','\n').strip()):
        paragraph=paragraph.strip()
        if not paragraph:continue
        kind='heading' if paragraph.startswith('# ') else 'paragraph'
        if kind=='heading':paragraph=paragraph[2:].strip()
        if paragraph.startswith('$$') and paragraph.endswith('$$'):
            kind='math';paragraph=paragraph[2:-2].strip()
        elif paragraph.startswith('$') and paragraph.endswith('$') and paragraph.count('$')==2:
            kind='math';paragraph=paragraph[1:-1].strip()
        if not paragraph:raise ValueError('Empty heading or equation')
        entries=[('math',paragraph)] if kind=='math' else tokens(paragraph)
        for source,word in entries:
            if source=='math':
                try:render_math(word)
                except Exception as error:raise ValueError(f'Equation {word!r}: {error}') from error
            elif not WORD.fullmatch(word):
                raise ValueError(f'Unsupported prose token {word!r}. Use English words with ASCII apostrophes/hyphens and one trailing punctuation mark; put numbers, symbols and equations inside $...$. Nothing has been omitted.')
        blocks.append({'kind':kind,'text':paragraph})
    return {'seed':settings.seed,'blocks':blocks,
            'layout':{'page':settings.page.lower(),'size_mm':list(PAGE_MM[settings.page]),'background':settings.background,
                      'dpi':settings.dpi,'margins_mm':list(settings.margins),'line_spacing_mm':settings.line_spacing,
                      'x_height_mm':settings.x_height,'word_height_mm':settings.x_height/.73,
                      'math_font_points':15*settings.x_height/3.1,
                      'word_spacing_mm':settings.word_spacing,'paragraph_extra_lines':settings.paragraph_gap,
                      'size_variation':settings.natural_variation,'vertical_variation_mm':.12,'heading_size_scale':1.25},
            'math':{'mode':'auto'},'processing':{'mode':'adaptive','opacity_target':settings.opacity_target}}

def expected_tokens(document):
    return [(source,text) for b in document['blocks'] for source,text in
            ([('math',b['text'])] if b['kind']=='math' else tokens(b['text']))]

def validate_layout(manifest):
    actual=[('word' if p['rendering']['source']=='diffusionpen_word' else 'math',p['text']) for p in manifest['positions']]
    if actual!=expected_tokens(manifest['document']):raise AssertionError('Input token order/count does not match rendered placements')
    left,top,right,bottom=manifest['geometry']['usable_box']
    for page in range(1,len(manifest['pages'])+1):
        active=[]
        for position in sorted((p for p in manifest['positions'] if p['page']==page),key=lambda p:p['box'][1]):
            x,y,r,b=position['box']
            if not left<=x<r<=right or not top<=y<b<=bottom:raise AssertionError('Content extends beyond page margins')
            active=[p for p in active if p['box'][3]>y]
            if any(x<p['box'][2] and r>p['box'][0] for p in active):raise AssertionError('Overlapping rendered content')
            active.append(position)

def destination(path):
    target=(ROOT/path).resolve()
    for directory in ('data','reforge','diffusionpen-training','training','.reforge','.venv','.git','.reforge-cache'):
        if target.is_relative_to((ROOT/directory).resolve()):raise ValueError('Choose an output folder outside protected data, models and source directories')
    return target

def friendly_error(error):
    detail=str(error)+'\n'+'\n'.join(getattr(error,'__notes__',[]))
    if isinstance(error,ModuleNotFoundError):return detail+'\nInstall the missing PDF additions with .\\.reforge\\Scripts\\python.exe -m pip install -r handwriting_app/requirements.txt; retain the existing model environment.'
    if isinstance(error,FileNotFoundError):return f'{detail}\nA required local file is missing. Check training/config.json, the checkpoint cache, style reference paths and output destination. Restore the existing files; no retraining is needed.'
    if 'out of memory' in str(error).lower():return f'{detail}\nClose other GPU tasks and retry with Device = cpu or fewer candidates. Completed words remain cached.'
    if isinstance(error,PermissionError):return f'{detail}\nChoose a writable destination; close software locking that file. Your original files were not overwritten.'
    return detail.strip()

class Runner:
    """One worker at a time; Reforge changes global RNG state and the working directory."""
    def __init__(self,pipeline_factory=Pipeline,store_factory=CandidateStore):
        self.pipeline_factory=pipeline_factory;self.store_factory=store_factory
        self.pipeline=None;self.lock=threading.Lock()

    def get_pipeline(self,settings):
        kwargs={} if settings.style_config=='page_engine/styles/erdem.json' else {'style_config':settings.style_config}
        candidate=self.pipeline_factory(settings.adapter,settings.device,**kwargs)
        if self.pipeline is None or candidate.identity!=self.pipeline.identity or candidate.device_option!=self.pipeline.device_option:
            self.pipeline=None  # Release the old model before loading a different one.
            self.pipeline=candidate
        return self.pipeline

    def generate(self,text,settings,output='outputs/app_runs',progress=lambda *a:None,cancel=None):
        if not self.lock.acquire(blocking=False):raise RuntimeError('A generation is already running; wait or cancel it first')
        try:
            cancel=cancel or threading.Event()
            def check():
                if cancel.is_set():raise Cancelled('Cancelled between words. Completed candidates remain cached.')
            check();progress(0,1,'Checking input and equations')
            document=parse_document(text,settings);check()
            folder=destination(output)/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            folder.mkdir(parents=True,exist_ok=False)
            (folder/'input.txt').write_text(text,encoding='utf-8');write_json(folder/'settings.json',asdict(settings))
            write_json(folder/'document.json',document)
            try:
                progress(0,1,'Verifying local pretrained checkpoints and shared style references')
                pipeline=self.get_pipeline(settings)
                sampling={'steps':settings.steps,'guidance':settings.guidance,'retries':0,'ocr':False}
                store=self.store_factory(pipeline,sampling,initial_count=settings.candidates)
                total=sum(s=='word' for s,t in expected_tokens(document))
                class ProgressStore:
                    processing=store.processing
                    def __init__(self):self.pipeline=pipeline;self.settings=store.settings;self.done=0;self.cached_words=0
                    @property
                    def hits(self):return self.cached_words
                    @property
                    def generated(self):return store.generated
                    def obtain(self,word,seed):
                        check();progress(self.done,max(total,1),f'Generating/reusing word {self.done+1}/{total}: {word}')
                        before=store.generated
                        result=store.obtain(word,seed)
                        if store.generated==before:self.cached_words+=1
                        self.done+=1;check()
                        progress(self.done,max(total,1),f'Prepared {self.done}/{total} words');return result
                manifest=build(document,ProgressStore(),str(folder/'preview'),read_json(ROOT/'page_engine/profiles/scanned.json'))
                check();validate_layout(manifest)
                session={'schema':1,'document':document,'model':pipeline.identity,'style_config':settings.style_config,
                         'adapter':settings.adapter,'sampling':sampling,'candidate_count':settings.candidates,'reuse_approved':False,
                         'words':[{'index':r['index'],'text':r['text'],'seed':r['seed'],'bundle_id':r['bundle_id']} for r in manifest['word_records']]}
                write_json(folder/'session.json',session)
                manifest['app']={'run_directory':str(folder),'session':str(folder/'session.json'),
                                 'input_preservation_verified':True,'layout_bounds_and_overlap_verified':True,
                                 'page_sha256':{path:sha(ROOT/path) for path in manifest['pages']}}
                write_json(folder/'preview.json',manifest)
                progress(total,max(total,1),f'Ready: {len(manifest["pages"])} pages; review spelling before exporting')
                return manifest
            except Exception as error:
                import traceback
                try:
                    (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
                    error.add_note('Diagnostics saved to '+str(folder/'error.log'))
                except OSError:pass
                raise
        finally:self.lock.release()

def export_pdf(manifest,output):
    """Embed the exact preview pixels losslessly; physical page size comes from layout."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader
    validate_layout(manifest)
    target=destination(output)
    if target.suffix.lower()!='.pdf':raise ValueError('Choose a .pdf filename')
    if target.exists():raise FileExistsError('Existing file preserved. Choose a new PDF filename: '+str(target))
    if not manifest['pages']:raise ValueError('Generate a preview first')
    mm=manifest['document']['layout'].get('size_mm',PAGE_MM.get(manifest['document']['layout']['page'].capitalize(),(210,297)))
    points=tuple(v*72/25.4 for v in mm);stream=io.BytesIO()
    pdf=canvas.Canvas(stream,pagesize=points,pageCompression=1)
    pdf.setTitle('Handwritten notes');pdf.setAuthor('Handwriting AI (local)')
    for path in manifest['pages']:
        expected_hash=manifest.get('app',{}).get('page_sha256',{}).get(path)
        if expected_hash and sha(ROOT/path)!=expected_hash:raise ValueError('Preview file changed; regenerate before exporting')
        with Image.open(ROOT/path) as image:
            image.load()
            if list(image.size)!=manifest['geometry']['size_pixels']:raise ValueError('Preview image dimensions changed')
            pdf.drawImage(ImageReader(image.convert('RGB')),0,0,width=points[0],height=points[1])
        pdf.showPage()
    pdf.save();data=stream.getvalue();reader=PdfReader(io.BytesIO(data))
    if len(reader.pages)!=len(manifest['pages']):raise AssertionError('PDF page count mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as file:file.write(data)
    return target
