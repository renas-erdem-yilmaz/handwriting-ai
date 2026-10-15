import streamlit as st

def pdf_download(snapshot):
    st.download_button('Download multipage PDF',snapshot['pdf'],file_name='handwritten_notes.pdf',mime='application/pdf',
                       key='pdf_'+snapshot['revision'],on_click='ignore',type='primary')

def png_download(snapshot,index):
    st.download_button(f'Download page {index+1} PNG',snapshot['png'][index],file_name=f'handwritten_page_{index+1:03d}.png',
                       mime='image/png',key=f'png_{snapshot["revision"]}_{index}',on_click='ignore')
