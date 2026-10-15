import streamlit as st
from page_engine.common import ROOT

SAMPLE=(ROOT/'handwriting_app/example.txt').read_text(encoding='utf-8')

def render(disabled=False):
    st.subheader('Write your notes')
    left,right=st.columns([1,2])
    with left:
        if st.button('Load sample',disabled=disabled):st.session_state['editor_text']=SAMPLE
    with right:
        uploaded=st.file_uploader('UTF-8 text file',type=['txt'],disabled=disabled)
        if st.button('Load uploaded text',disabled=disabled or uploaded is None):
            try:
                raw=uploaded.getvalue()
                if len(raw)>500_000:raise ValueError('Split files larger than 500 KB into smaller notes')
                st.session_state['editor_text']=raw.decode('utf-8-sig')
            except (ValueError,UnicodeError) as error:st.error('Cannot load text: '+str(error))
    text=st.text_area('Notes',key='editor_text',height=330,disabled=disabled)
    st.caption('Blank lines separate paragraphs. Start a heading with `# Title`. Use `$v_0+at$` inline, or a separate `$F = ma$` paragraph for an equation.')
    return text
