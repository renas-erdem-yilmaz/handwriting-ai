"""Serialized background jobs, review integration and immutable downloadable snapshots."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
import hashlib
import io
import json
import re
import threading
import traceback
import uuid
from PIL import Image
from page_engine.common import ROOT, read_json, sha
from page_engine.layout import build
from handwriting_app.service import Runner, Settings, Cancelled, destination, validate_layout, export_pdf, write_json, friendly_error

DEFAULT_STYLE='page_engine/styles/erdem.json'

def style_profile(references):
    original=read_json(ROOT/DEFAULT_STYLE)
    if len(references)!=5 or len(set(references))!=5:raise ValueError('Choose exactly five different reference word images')
    if set(references)==set(original['references']):return DEFAULT_STYLE,None
    references=sorted(references)
    for ref in references:
        path=(ROOT/ref).resolve()
        if not path.is_relative_to((ROOT/'data/finetune_words').resolve()):raise ValueError('Choose an existing prepared reference word')
        with Image.open(path) as image:image.verify()
    data={'name':'User-selected five reference words','references':references,'ink_opacity_target':.82}
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]
    return '.reforge-cache/app_styles/'+digest+'.json',data

def public_error(error):
    if isinstance(error,FileNotFoundError):
        name=Path(error.filename).name if error.filename else 'a required local file'
        return f'Missing {name}. Restore the existing model checkpoints, reference images or generated cache files, then retry. No retraining is needed.'
    message=re.sub(r'Diagnostics saved to [^\n]*','Diagnostic details were saved with this run.',friendly_error(error)).strip()
    for prefix in (str(ROOT),str(ROOT).replace('\\','/')):message=message.replace(prefix,'<project>')
    return message

def png_bytes(path,dpi):
    with Image.open(ROOT/path) as image:
        stream=io.BytesIO();image.save(stream,format='PNG',dpi=(dpi,dpi));return stream.getvalue()

@dataclass
class Job:
    kind: str
    cancel: threading.Event=field(default_factory=threading.Event)
    lock: threading.Lock=field(default_factory=threading.Lock)
    progress: tuple=(0,1,'Starting')
    done: bool=False
    result: object=None
    error: str|None=None
    def update(self,*progress):
        with self.lock:self.progress=progress
    def state(self):
        with self.lock:return self.done,self.result,self.error,self.progress

class Backend:
    """Cache this one resource in Streamlit. Workers never call Streamlit APIs."""
    def __init__(self,runner=None,output='outputs/streamlit_runs'):
        self.runner=runner or Runner();self.output=output;self.gate=threading.Lock()

    def start(self,kind,operation):
        if not self.gate.acquire(blocking=False):raise RuntimeError('Another document/model task is running. Wait for it to finish, then retry.')
        job=Job(kind)
        def worker():
            try:
                result=operation(job)
                with job.lock:job.result=result
            except Exception as error:
                with job.lock:job.error=public_error(error)
                folder=destination(self.output)/('failed_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True,exist_ok=True)
                (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            finally:
                self.gate.release()
                with job.lock:job.done=True
        threading.Thread(target=worker,daemon=True,name='handwriting-model-worker').start()
        return job

    def generate(self,text,settings,profile_data=None):
        def operation(job):
            if profile_data is not None:write_json(ROOT/settings.style_config,profile_data)
            manifest=self.runner.generate(text,settings,self.output,job.update,job.cancel)
            return self.package(manifest,text,settings,job)
        return self.start('generate',operation)

    def package(self,manifest,text,settings,job):
        job.update(1,1,'Preparing lossless PNG and PDF downloads')
        folder=ROOT/manifest['app']['run_directory']
        pdf=export_pdf(manifest,folder/'document.pdf')
        return {'manifest':manifest,'text':text,'settings':asdict(settings),'pdf':pdf.read_bytes(),
                'png':[png_bytes(path,manifest['document']['layout']['dpi']) for path in manifest['pages']],
                'revision':folder.name}

    def session_store(self,snapshot):
        session=read_json(snapshot['manifest']['app']['session'])
        pipeline=SimpleNamespace(identity=session['model'],style_config={'ink_opacity_target':snapshot['settings'].get('opacity_target',.82)})
        store=self.runner.store_factory(pipeline,session['sampling'],initial_count=1)
        return session,store

    def bundle(self,snapshot,index):
        session,store=self.session_store(snapshot)
        item=next(word for word in session['words'] if word['index']==index)
        bundle=store.load(item['bundle_id'])
        if bundle['request']['model']!=session['model'] or bundle['request']['text']!=item['text'] or bundle['request']['seed']!=item['seed']:
            raise ValueError('Word review bundle no longer matches this preview')
        return bundle

    def recompose(self,snapshot,job):
        session,store=self.session_store(snapshot);document=session['document']
        lookup={(item['text'],item['seed']):item['bundle_id'] for item in session['words']}
        class CachedOnly:
            pipeline=store.pipeline;settings=store.settings;processing=document.get('processing',{});hits=0;generated=0
            def obtain(inner,text,seed):
                if job.cancel.is_set():raise Cancelled('Page refresh cancelled; original preview remains available')
                bundle=store.load(lookup[(text,seed)])
                if bundle['request']['model']!=session['model'] or bundle['request']['sampling']!=session['sampling']:
                    raise ValueError('Cached word model/sampling mismatch')
                selected=bundle['selected'] if bundle['selected'] is not None else 0
                chosen=bundle['candidates'][selected];path=ROOT/chosen['image']
                if sha(path)!=chosen['sha256']:raise ValueError('Cached word image changed')
                with Image.open(path) as image:raw=image.convert('L')
                flags=list(chosen['flags'])+(['manually_problematic'] if bundle['status']=='problematic' else [])
                inner.hits+=1
                return raw,{'id':chosen['cache_id'],'bundle_id':bundle['id'],'candidate_index':selected,'status':bundle['status'],'flags':flags}
        folder=destination(self.output)/('review_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True,exist_ok=False)
        job.update(0,1,'Refreshing page layout from existing cached words; no unrelated inference')
        manifest=build(document,CachedOnly(),str(folder/'preview'),snapshot['manifest']['profile'],
                       {str(item['index']):item['seed'] for item in session['words']})
        validate_layout(manifest)
        write_json(folder/'session.json',session);write_json(folder/'settings.json',snapshot['settings'])
        (folder/'input.txt').write_text(snapshot['text'],encoding='utf-8')
        manifest['app']={'run_directory':str(folder),'session':str(folder/'session.json'),
                         'input_preservation_verified':True,'layout_bounds_and_overlap_verified':True,
                         'page_sha256':{p:sha(ROOT/p) for p in manifest['pages']}}
        write_json(folder/'preview.json',manifest)
        return self.package(manifest,snapshot['text'],Settings(**snapshot['settings']),job)

    def refresh(self,snapshot):return self.start('refresh',lambda job:self.recompose(snapshot,job))

    def choose(self,snapshot,index,candidate,status='approved',note=''):
        def operation(job):
            session,store=self.session_store(snapshot)
            item=next(word for word in session['words'] if word['index']==index)
            store.decide(item['bundle_id'],candidate,status,note)
            return self.recompose(snapshot,job)
        return self.start('review',operation)

    def regenerate(self,snapshot,index,seed=None):
        def operation(job):
            settings=Settings(**snapshot['settings'])
            job.update(0,1,'Checking model and generating one additional word candidate')
            pipeline=self.runner.get_pipeline(settings)
            session,_=self.session_store(snapshot)
            if pipeline.identity!=session['model']:raise ValueError('Model/reference hashes changed. Generate a new document before regenerating this word.')
            store=self.runner.store_factory(pipeline,session['sampling'],initial_count=1)
            item=next(word for word in session['words'] if word['index']==index)
            if job.cancel.is_set():raise Cancelled('Candidate regeneration cancelled')
            bundle=store.append(item['bundle_id'],seed)
            job.update(1,1,'Candidate saved. Select and approve it to update the page.')
            return {'index':index,'bundle':bundle}
        return self.start('candidate',operation)
