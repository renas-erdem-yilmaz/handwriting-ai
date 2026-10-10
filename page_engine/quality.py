"""Prepare persistent candidates, render reviewed pages, or launch local review."""
import argparse
from .common import ROOT, read_json, write_json
from .model import Pipeline
from .candidates import CandidateStore
from .layout import tokens, build

def open_store(session):
    pipeline = Pipeline(session.get('adapter'), style_config=session['style_config'])
    if pipeline.identity != session['model']: raise ValueError('Model/reference hashes changed; prepare a new session')
    return CandidateStore(pipeline, session['sampling'], initial_count=session['candidate_count'], reuse_approved=session.get('reuse_approved',False))

def prepare(document_path, session_path, style_config, steps=20, guidance=3., count=3, adapter=None, ocr=False, reuse_approved=False):
    document = read_json(ROOT / document_path)
    pipeline = Pipeline(adapter, style_config=style_config)
    settings = {'steps':steps, 'guidance':guidance, 'retries':0, 'ocr':ocr}
    store = CandidateStore(pipeline, settings, initial_count=count, reuse_approved=reuse_approved)
    entries = []; index = 0
    for block in document['blocks']:
        if block['kind'] == 'math': continue
        for source, text in tokens(block['text']):
            if source != 'word': continue
            seed = document.get('word_seeds',{}).get(str(index),document.get('seed',42) + index*1009)
            bundle_id, _ = store.key(text, seed)
            entries.append({'index':index, 'text':text, 'seed':seed, 'bundle_id':bundle_id}); index += 1
    session = {'schema':1, 'document':document, 'model':pipeline.identity, 'style_config':style_config,
               'adapter':adapter, 'sampling':settings, 'candidate_count':count, 'words':entries, 'reuse_approved':reuse_approved}
    path = ROOT / session_path
    if path.exists():
        if read_json(path) != session: raise ValueError('Existing session differs; choose a new session path')
    else: write_json(path, session)
    for n, item in enumerate(entries,1):
        print(f'Candidates [{n}/{len(entries)}] {item["text"]}',flush=True)
        store.ensure(item['text'],item['seed'])
    return session, store

def render_session(session, store, output, require_approved=False, legacy=False):
    doc = dict(session['document'])
    doc['processing'] = dict(doc.get('processing',{}), mode='legacy' if legacy else 'adaptive')
    if legacy: doc['layout'] = dict(doc.get('layout',{}), size_variation=.03, vertical_variation_mm=.17)
    return build(doc, store, output, read_json(ROOT/'page_engine/profiles/scanned.json'),
                 {str(item['index']):item['seed'] for item in session['words']}, require_approved)

def main():
    p = argparse.ArgumentParser(description='Step 9.5 candidate review and conservative consistency tools')
    sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--document',required=True); prep.add_argument('--session',required=True)
    prep.add_argument('--style',default='page_engine/styles/erdem.json')
    prep.add_argument('--steps',type=int,default=20); prep.add_argument('--guidance',type=float,default=3.)
    prep.add_argument('--candidates',type=int,default=3,choices=[1,2,3]); prep.add_argument('--adapter')
    prep.add_argument('--ocr',action='store_true'); prep.add_argument('--reuse-approved',action='store_true')
    render = sub.add_parser('render'); render.add_argument('--session',required=True); render.add_argument('--output',required=True)
    render.add_argument('--require-approved',action='store_true'); render.add_argument('--legacy',action='store_true')
    gui = sub.add_parser('review'); gui.add_argument('--session',required=True)
    regen = sub.add_parser('regenerate'); regen.add_argument('--session',required=True); regen.add_argument('--word-index',type=int,required=True)
    regen.add_argument('--seed',type=int)
    args = p.parse_args()
    if args.command == 'prepare':
        if not 1<=args.steps<=200 or not 0<args.guidance<=20: raise ValueError('Invalid sampling bounds')
        session,store = prepare(args.document,args.session,args.style,args.steps,args.guidance,args.candidates,args.adapter,args.ocr,args.reuse_approved)
        print('Session saved:',args.session,'new candidates:',store.generated); return
    session = read_json(ROOT / args.session); store = open_store(session)
    if args.command == 'review':
        from .review_ui import launch
        launch(session,store); return
    if args.command == 'regenerate':
        item = next(item for item in session['words'] if item['index']==args.word_index)
        bundle = store.append(item['bundle_id'],args.seed)
        print('Added candidate; existing selection retained:',bundle['selected']); return
    result = render_session(session,store,args.output,args.require_approved,args.legacy)
    print('Pages:',result['pages'],'new candidates:',store.generated)

if __name__ == '__main__': main()
