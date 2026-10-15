import csv
import streamlit as st
from page_engine.common import ROOT,read_json
from handwriting_app.service import Settings
from .backend import DEFAULT_STYLE,style_profile

def reference_catalog():
    records={}
    for split in ('train','val','test'):
        path=ROOT/'data/finetune_words'/split/'labels.csv'
        if not path.exists():continue
        with path.open(encoding='utf-8-sig',newline='') as file:
            for row in csv.DictReader(file):
                relative=f'data/finetune_words/{split}/images/'+row['filename']
                if (ROOT/relative).is_file():records[relative]=f'{row["text"]} · {split} sample {len(records)+1}'
    return records

def render(disabled=False):
    with st.sidebar:
        st.header('Handwriting settings')
        choices=['Pretrained DiffusionPen']
        if (ROOT/'outputs/finetune/erdem_pilot/last_adapter.pt').exists():choices.append('Pilot style adapter (30 updates)')
        model=st.selectbox('Model',choices,key='model',disabled=disabled)
        style=st.radio('Style references',['Existing Erdem style','Choose five words'],key='style_mode',disabled=disabled)
        default=read_json(ROOT/DEFAULT_STYLE)['references']
        catalog=reference_catalog();profile_data=None;style_path=DEFAULT_STYLE;style_error=None
        if style=='Choose five words':
            selected=st.multiselect('Five distinct reference words',list(catalog),default=[p for p in default if p in catalog],
                                    format_func=lambda p:catalog[p],key='references',disabled=disabled)
            try:style_path,profile_data=style_profile(selected)
            except Exception as error:style_error=str(error);st.warning(style_error)
        else:selected=default
        with st.expander('View selected reference handwriting'):
            for ref in selected:
                if (ROOT/ref).exists():st.image(str(ROOT/ref),caption=catalog.get(ref,'Existing handwriting reference'))
        page=st.selectbox('Page size',['A4','Notebook'],key='page_size',disabled=disabled)
        background=st.selectbox('Paper background',['blank','lined'],key='paper',disabled=disabled)
        size=st.slider('Writing x-height (mm)',1.5,6.0,3.1,.1,key='writing_size',disabled=disabled)
        spacing=st.slider('Word spacing (mm)',0.,10.,2.4,.2,key='word_gap',disabled=disabled)
        pitch=st.slider('Line spacing (mm)',5.,25.,8.,.5,key='line_gap',disabled=disabled)
        paragraph=st.slider('Paragraph gap (lines)',0.,4.,.8,.1,key='paragraph_gap',disabled=disabled)
        darkness=st.slider('Ink darkness',.55,.95,.82,.01,key='ink',disabled=disabled)
        variation=st.slider('Natural size variation (%)',0.,5.,2.5,.5,key='variation',disabled=disabled)/100
        with st.expander('Margins and sampling'):
            margins=tuple(st.number_input(f'{name} margin (mm)',min_value=6.,max_value=70.,value=float(value),step=1.,key='margin_'+name,disabled=disabled)
                          for name,value in zip(('Left','Right','Top','Bottom'),(16,16,22,18)))
            seed=st.number_input('Random seed',min_value=0,max_value=2**32-1,value=9500,step=1,key='seed',disabled=disabled)
            steps=st.selectbox('Sampling steps',[20,30,50],key='steps',disabled=disabled)
            candidates=st.selectbox('Initial candidates per word',[1,2,3],key='candidate_count',disabled=disabled)
            device=st.selectbox('Device',['auto','cuda','cpu'],key='device',disabled=disabled)
        st.caption('One shared reference set per document. Darkness changes are bounded; variation is limited to ±5%. The pilot adapter has no established quality improvement.')
    settings=Settings(page=page,background=background,x_height=size,word_spacing=spacing,line_spacing=pitch,
        paragraph_gap=paragraph,opacity_target=darkness,natural_variation=variation,margins=margins,seed=int(seed),steps=steps,
        candidates=candidates,device=device,style_config=style_path,
        adapter=None if model==choices[0] else 'outputs/finetune/erdem_pilot/last_adapter.pt')
    return settings,profile_data,style_error
