"""Real offline pretrained inference plus a multi-page physics PDF sample."""
import time
from .service import Runner,Settings,export_pdf
from .gui import SAMPLE
from page_engine.common import ROOT,read_json,write_json,sha
from pypdf import PdfReader

def main():
    output=ROOT/'outputs/app_tests'
    if (output/'integration_report.json').exists():raise FileExistsError('Integration outputs already exist; preserve them')
    protected=read_json(ROOT/'outputs/quality_tests/protected_before.json')
    recent={str(p.relative_to(ROOT)):sha(p) for name in ('quality_tests','size_tests','math_tests') for p in (ROOT/'outputs'/name).rglob('*') if p.is_file()}
    expected_model=read_json(ROOT/'outputs/quality_tests/after_final.json')['model']
    runner=Runner();start=time.perf_counter()
    progress=lambda n,total,message:print(message,flush=True)
    import torch
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    real=runner.generate('# Physics\n\n$F = ma$',Settings(seed=109601),str(output/'real_smoke'),progress)
    if real['new_words']<1:raise AssertionError('Real smoke seed already cached; no fresh inference occurred')
    assert runner.pipeline.ready and runner.pipeline.identity==expected_model
    real_pdf=export_pdf(real,str(output/'real_inference.pdf'))
    # Existing exemplar text with the application's default seeds; wider pitch demonstrates pagination.
    sample=runner.generate(SAMPLE,Settings(line_spacing=18),str(output/'sample'),progress)
    if len(sample['pages'])<2:raise AssertionError('Multi-page sample did not paginate')
    sample_pdf=export_pdf(sample,str(output/'handwritten_notes.pdf'))
    changed=[name for name,digest in {**protected,**recent}.items() if sha(ROOT/name)!=digest]
    if changed:raise AssertionError('Protected files changed: '+str(changed))
    report={'step':10,'real_model_forward_inference_succeeded':True,'device':runner.pipeline.device,
            'checkpoint_identity':runner.pipeline.identity,'uncached_smoke_candidates':real['new_words'],
            'real_smoke_manifest':str((ROOT/real['app']['run_directory']/'preview.json').relative_to(ROOT)),
            'real_smoke_pdf':str(real_pdf.relative_to(ROOT)),'sample_pdf':str(sample_pdf.relative_to(ROOT)),
            'sample_pages':sample['pages'],'sample_pdf_page_count':len(PdfReader(sample_pdf).pages),
            'sample_cache_hits':sample['cache_hits'],'sample_new_candidates':sample['new_words'],
            'duration_seconds':round(time.perf_counter()-start,2),
            'peak_allocated_gpu_gib':round(torch.cuda.max_memory_allocated()/2**30,3) if torch.cuda.is_available() else None,
            'protected_files_unchanged':len({**protected,**recent}),'math_font_fallback':True,
            'limitations':['No handwritten math symbols have been approved; all equations are labeled font fallback.',
                           'Pretrained prose candidates retain model spelling/shape limitations and pending review status.',
                           'The multi-page sample intentionally uses 18 mm line pitch to demonstrate page navigation/export.']}
    write_json(output/'integration_report.json',report)
    print('Integration successful:',report['sample_pdf'],'pages:',report['sample_pdf_page_count'],flush=True)

if __name__=='__main__':main()
