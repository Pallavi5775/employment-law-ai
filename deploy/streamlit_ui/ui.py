# deploy/streamlit_ui/ui.py
import streamlit as st
import requests
import base64
# st.secrets.get('API_URL','http://localhost:8080')
API_URL = 'http://localhost:8080'

st.title('Employment Contract Intelligence — Demo UI')

uploaded = st.file_uploader('Upload contract (PDF/DOCX/TXT)', type=['pdf','docx','txt'])
if uploaded is not None:
    # naive text extraction for demo; in production use OCR and robust parsers
    try:
        content = uploaded.read().decode('utf-8', errors='ignore')[:100000]
    except Exception:
        content = 'Could not extract text from file in this demo. Upload a plain text file to test.'
    st.text_area('Raw text (truncated)', content, height=200)
    if st.button('Extract Clauses'):
        resp = requests.post(f"{API_URL}/extract", json={'text': content}).json()
        st.write('### Clauses detected')
        for c in resp.get('clauses', []):
            st.write(f"- **{c['clause']}** (score={c['score']}) -> {c['span']}")
        st.write('### Summary')
        st.write(resp.get('summary', ''))

    if st.button('Predict Risk'):
        resp = requests.post(f"{API_URL}/predict_risk", json={'text': content}).json()
        st.metric('Risk score', resp.get('risk_score',0))
