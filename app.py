"""Handwriting AI 1.0 — localhost Streamlit entry point, no implicit inference."""
from dataclasses import asdict
import streamlit as st
from app_components.resources import get_backend
from app_components import editor,settings,preview,review
from app_components.backend import public_error

st.set_page_config(page_title='Handwriting AI',page_icon='✍️',layout='wide')
st.title('Handwriting AI')
st.caption('Personal handwriting · local generation · notes to PDF')
st.session_state.setdefault('editor_text',editor.SAMPLE)
st.session_state.setdefault('snapshot',None)
st.session_state.setdefault('job',None)
st.session_state.setdefault('word_selections',{})
backend=get_backend()
job=st.session_state['job'];busy=job is not None and not job.state()[0]
configuration,profile_data,style_error=settings.render(busy)
st.session_state['current_settings']=asdict(configuration)

@st.fragment(run_every='1s' if busy else None)
def progress_panel():
    job=st.session_state['job']
    if job is None:return
    done,result,error,progress=job.state()
    amount,total,message=progress
    st.progress(min(1.,amount/max(1,total)),text=message)
    if not done:
        if st.button('Cancel after current operation',key='cancel_job'):job.cancel.set();st.info('Cancellation requested; the current word call must finish first.')
    else:
        if error:st.session_state['last_error']=error
        elif job.kind=='candidate':st.session_state['last_notice']='New candidate saved. Select it in Word review to update the page.'
        else:st.session_state['snapshot']=result;st.session_state['last_notice']='Pages and downloads are ready. Review spelling before use.'
        st.session_state['job']=None
        st.rerun()

progress_panel()
if st.session_state.get('last_error'):st.error(st.session_state['last_error'])
if st.session_state.get('last_notice'):st.success(st.session_state['last_notice'])
tabs=st.tabs(['Editor','Preview & export','Word review','Mathematical symbols'])
with tabs[0]:
    text=editor.render(busy)
    if st.button('Generate handwriting',type='primary',disabled=busy or style_error is not None,key='generate'):
        try:
            configuration.validate()
            st.session_state['last_error']=None;st.session_state['last_notice']=None
            st.session_state['job']=backend.generate(text,configuration,profile_data);st.rerun()
        except Exception as error:st.error(public_error(error))
    old=st.session_state['snapshot']
    if old and (old['text']!=text or old['settings']!=asdict(configuration)):
        st.info('Editor/settings differ from the displayed snapshot. Nothing regenerates automatically. Press Generate to apply these changes; downloads still match the displayed pages.')
with tabs[1]:
    preview.render(st.session_state['snapshot'])
    if st.session_state['snapshot'] and st.button('Refresh pages from saved word choices / math library',disabled=busy,key='refresh'):
        try:st.session_state['job']=backend.refresh(st.session_state['snapshot']);st.rerun()
        except Exception as error:st.error(public_error(error))
with tabs[2]:
    try:review.render(st.session_state['snapshot'],backend,busy)
    except Exception as error:st.error(public_error(error))
with tabs[3]:
    st.subheader('Personal mathematical symbols')
    st.write('Only reviewed and approved symbol crops are eligible. An incomplete inventory uses the original mathematical font renderer with explicit fallback labels.')
    st.code(r'.\.reforge\Scripts\python.exe -m page_engine.math_symbols collect',language='powershell')
    st.write('From the project folder: open a scanned page, crop one symbol, enter its exact label, review the crop/scale/baseline, and explicitly approve it. Multiple variants are supported. Then refresh the pages.')
    st.caption('Examples: F = ma; a = F/m; v = v_0 + at; x = 1/2 at^2; θ = α + β. Use braces for compound scripts and parentheses for denominators: v^{12}, F/(ma). No general LaTeX parsing or automatic symbol approval is attempted.')
