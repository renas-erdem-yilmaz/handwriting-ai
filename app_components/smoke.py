"""Real Streamlit controls → CUDA inference → pending review → notebook PDF/PNG."""
import io
import time
from unittest.mock import patch
from PIL import Image
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from page_engine.common import ROOT,read_json,sha
from handwriting_app.service import write_json
from .resources import get_backend
from .backend import Backend

def await_ui(app,timeout=240):
    deadline=time.monotonic()+timeout
    while app.session_state['job'] is not None and time.monotonic()<deadline:
        time.sleep(.2);app.run(timeout=30)
        if app.exception:raise AssertionError([e.message for e in app.exception])
    if app.session_state['job'] is not None:raise AssertionError('Real generation timed out')
    if app.session_state.get('last_error'):raise AssertionError(app.session_state['last_error'])

def main():
    parent=ROOT/'outputs/app_tests/streamlit_v1'
    if (parent/'real_integration.json').exists():raise FileExistsError('Previous real integration report is preserved')
    protected=read_json(ROOT/'outputs/quality_tests/protected_before.json')
    protected.update({str(p.relative_to(ROOT)):sha(p) for name in ('quality_tests','size_tests','math_tests')
                      for p in (ROOT/'outputs'/name).rglob('*') if p.is_file()})
    # Also preserve all existing word selections. Only newly generated test bundles are edited.
    decisions={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'.reforge-cache/candidate_reviews').glob('*.json')}
    parent.mkdir(parents=True,exist_ok=True);backend=Backend(output=str(parent/'real_runs'))
    import torch
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();get_backend.clear()
    # Only the output resource location is injected; Runner/Pipeline/inference are real.
    with patch('app_components.resources.Backend',return_value=backend):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        assert not app.exception and backend.runner.pipeline is None
        app.text_area(key='editor_text').set_value('# Physics\n\n$F = ma$\n\n$θ = α + β$')
        app.number_input(key='seed').set_value(1012030)
        app.button(key='generate').click().run();await_ui(app)
        first=app.session_state['snapshot']
        assert first['manifest']['new_words']>=1 and backend.runner.pipeline.ready
        pipeline_id=id(backend.runner.pipeline)
        app.button(key='regenerate_word').click().run();await_ui(app)
        bundle=backend.bundle(first,0);assert len(bundle['candidates'])==2 and bundle['status']=='pending'
        app.radio(key='pick_'+bundle['id']).set_value(1).run()
        app.button(key='select_word').click().run();await_ui(app)
        reviewed=app.session_state['snapshot']
        assert reviewed['manifest']['new_words']==0 and backend.bundle(reviewed,0)['status']=='pending'
        assert backend.bundle(reviewed,0)['selected']==1
        app.button[0].click().run()  # Load sample; never implicitly generates.
        app.number_input(key='seed').set_value(9500)
        app.selectbox(key='page_size').set_value('Notebook')
        app.selectbox(key='paper').set_value('lined')
        app.slider(key='line_gap').set_value(18.)
        app.button(key='generate').click().run();await_ui(app)
        notebook=app.session_state['snapshot']
        assert len(notebook['png'])>=2 and notebook['manifest']['new_words']==0
        assert notebook['manifest']['cache_hits']==32 and id(backend.runner.pipeline)==pipeline_id
        # A4 sample uses identical stored word candidates, ordinary spacing and no extra inference.
        app.selectbox(key='page_size').set_value('A4');app.slider(key='line_gap').set_value(8.)
        app.button(key='generate').click().run();await_ui(app)
        a4=app.session_state['snapshot'];assert a4['manifest']['new_words']==0
        before=a4['pdf'];app.select_slider(key='preview_zoom').set_value('Enlarge').run();app.run()
        assert app.session_state['snapshot']['pdf']==before and not app.exception
    samples={}
    for name,snapshot in [('real_word',reviewed),('notebook',notebook),('a4',a4)]:
        folder=parent/name;folder.mkdir(exist_ok=False)
        (folder/'document.pdf').write_bytes(snapshot['pdf']);write_json(folder/'manifest.json',snapshot['manifest'])
        reader=PdfReader(io.BytesIO(snapshot['pdf']));assert len(reader.pages)==len(snapshot['png'])
        for n,(data,path,page) in enumerate(zip(snapshot['png'],snapshot['manifest']['pages'],reader.pages),1):
            (folder/f'page_{n:03d}.png').write_bytes(data)
            objects=[v.get_object() for v in page['/Resources']['/XObject'].values() if v.get_object()['/Subtype']=='/Image']
            with Image.open(io.BytesIO(data)) as image,Image.open(ROOT/path) as raw:
                assert image.tobytes()==raw.tobytes() and objects[0].get_data()==image.convert('RGB').tobytes()
                assert abs(image.info['dpi'][0]-150)<.02
        samples[name]={'pdf':str((folder/'document.pdf').relative_to(ROOT)),'pages':len(snapshot['png'])}
    assert all(sha(ROOT/name)==digest for name,digest in {**protected,**decisions}.items())
    report={'real_streamlit_ui_to_pretrained_inference_passed':True,'startup_loaded_no_model':True,
            'device':backend.runner.pipeline.device,'model_identity':backend.runner.pipeline.identity,
            'fresh_initial_candidates':first['manifest']['new_words'],'individual_word_regeneration_passed':True,
            'human_approvals_performed':False,'existing_approvals_and_protected_files_unchanged':len({**protected,**decisions}),
            'cache_only_word_choice_refresh_passed':True,'notebook_cache_hits':notebook['manifest']['cache_hits'],
            'no_duplicate_model_instance_across_jobs':True,'rerun_preserved_pdf':True,
            'png_dpi_and_pdf_pixel_identity_verified':True,'samples':samples,
            'duration_seconds':round(time.perf_counter()-start,2),
            'peak_gpu_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3) if torch.cuda.is_available() else None,
            'math_fallback':'No approved glyph library; equations use labeled fonts.',
            'test_method':'Actual Streamlit AppTest widget actions; real pretrained pipeline and GPU, not mocked generation.'}
    write_json(parent/'real_integration.json',report)
    print('Real Streamlit inference, regeneration, cached notebook/A4 generation, PNG/PDF export passed.',flush=True)

if __name__=='__main__':main()
