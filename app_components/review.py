import streamlit as st
from PIL import Image
from page_engine.common import ROOT,sha

def render(snapshot,backend,busy=False):
    if snapshot is None:st.info('Generate a document before reviewing words.');return
    records=snapshot['manifest']['word_records']
    if not records:st.info('This document contains equations only.');return
    st.subheader('Review generated words')
    only=st.checkbox('Show only flagged/problematic words',value=False,key='flagged_only')
    current=[]
    for item in records:
        flagged=item['status']=='problematic' or bool(item['flags'])
        if not only or flagged:current.append((item,flagged))
    if not current:st.success('No structural flags. This does not establish spelling accuracy.');return
    lookup={item['index']:(item,flagged) for item,flagged in current}
    index=st.selectbox('Word occurrence',list(lookup),format_func=lambda n:f'{n+1}. {lookup[n][0]["text"]}'+(' — flagged' if lookup[n][1] else ''),key='review_word')
    item,flagged=lookup[index];bundle=backend.bundle(snapshot,index)
    st.write(f'Requested word: **{item["text"]}** · Review status: **{bundle["status"]}**')
    columns=st.columns(3)
    for n,candidate in enumerate(bundle['candidates']):
        path=ROOT/candidate['image']
        if sha(path)!=candidate['sha256']:st.error('Candidate image integrity check failed.');return
        with columns[n%3]:
            with Image.open(path) as image:
                image=image.convert('RGB');factor=min(3,320/image.width,140/image.height)
                image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.Resampling.LANCZOS)
                st.image(image,caption=f'Candidate {n+1} · seed {candidate["seed"]}',width='content')
            st.caption(', '.join(candidate['flags']) or 'No structural flag; spelling unverified')
    chosen=st.radio('Preferred candidate',list(range(len(bundle['candidates']))),index=bundle['selected'] or 0,
                    format_func=lambda n:f'Candidate {n+1}',horizontal=True,key='pick_'+bundle['id'])
    note=st.text_input('Review note',value=bundle.get('problem_note',''),key='note_'+bundle['id'])
    left,middle,right=st.columns(3)
    with left:
        if st.button('Use and approve',disabled=busy,key='approve_word'):
            st.session_state['job']=backend.choose(snapshot,index,chosen,'approved',note)
            st.session_state.setdefault('word_selections',{})[bundle['id']]=chosen
            st.rerun()
    with middle:
        if st.button('Use without approval',disabled=busy,key='select_word'):
            st.session_state['job']=backend.choose(snapshot,index,chosen,'pending',note);st.rerun()
    with right:
        if st.button('Mark problematic',disabled=busy,key='problem_word'):
            st.session_state['job']=backend.choose(snapshot,index,bundle['selected'],'problematic',note);st.rerun()
    if st.button('Generate another candidate for this word',disabled=busy,key='regenerate_word'):
        st.session_state['job']=backend.regenerate(snapshot,index);st.rerun()
    st.caption('Regeneration appends one candidate and preserves prior selections. Choosing a candidate refreshes layout from cache; other words are not inferred again. Approvals persist on disk.')
