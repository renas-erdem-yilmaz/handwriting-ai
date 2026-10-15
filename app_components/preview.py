import streamlit as st
from .export import pdf_download,png_download

def render(snapshot):
    if snapshot is None:st.info('Generate a document to see its pages. No inference runs until you request it.');return
    st.subheader('Generated pages')
    manifest=snapshot['manifest']
    st.caption(f'{len(manifest["pages"])} pages · {manifest["cache_hits"]} cached word occurrences · {manifest["new_words"]} new candidates · {snapshot["settings"]["dpi"]} DPI')
    pdf_download(snapshot)
    zoom=st.select_slider('Preview size',['Fit','Enlarge','Full resolution'],value='Fit',key='preview_zoom')
    width={'Fit':550,'Enlarge':900,'Full resolution':'content'}[zoom]
    for index,data in enumerate(snapshot['png']):
        st.markdown(f'**Page {index+1}**')
        st.image(data,width=width)
        png_download(snapshot,index)
    fallback=[p for p in manifest['positions'] if p['rendering']['source']=='typeset_math_not_handwriting']
    if fallback:
        missing=sorted({s for p in fallback for s in p['rendering'].get('missing_symbols',[])})
        st.warning('Mathematical font fallback is present and is not personal handwriting. Missing approved symbols: '+(', '.join(missing) or 'expression outside the restricted handwritten syntax')+'.')
    st.caption('Preview and downloads use the same page pixels. Generated handwriting still requires a spelling/readability review.')
