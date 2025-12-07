# deploy/inference_api/app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn, os, json
import torch, numpy as np
from training.train_clause_classifier import LABELS

app = FastAPI(title='Employment Contract Intelligence API')

class ExtractRequest(BaseModel):
    text: str

@app.post('/extract')
async def extract(payload: ExtractRequest):
    # This is a lightweight stub. Replace with real model inference.
    text = payload.text.lower()
    results = []
    if 'confidential' in text or 'nondisclos' in text:
        results.append({'clause':'CONFIDENTIALITY','span':'confidential','score':0.95})
    if 'non-compete' in text or 'non compete' in text:
        results.append({'clause':'NON_COMPETE','span':'non-compete','score':0.9})
    if 'severance' in text or 'severance pay' in text:
        results.append({'clause':'SEVERANCE','span':'severance','score':0.85})
    if 'notice period' in text or 'notice' in text:
        results.append({'clause':'NOTICE_PERIOD','span':'notice period','score':0.8})
    return {'clauses': results, 'summary':'Detected {} clauses'.format(len(results))}

@app.post('/predict_risk')
async def predict_risk(payload: ExtractRequest):
    # stub returning rule-based risk
    score = 0
    txt = payload.text.lower()
    if 'severance' in txt:
        score += 10
    if 'no liability' in txt:
        score += 40
    if 'non-compete' in txt:
        score += 30
    return {'risk_score': score}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT',8080)))
