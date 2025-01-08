import numpy as np
import torch
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from flask import request
import joblib
import sklearn
import os
from mangum import Mangum
from fastapi import Body
app = FastAPI()
handler = Mangum(app)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base")

model = joblib.load("model_logistic_regression_detect_toxic.h5")



def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = phobert(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

text_temp = "Hello xin chào"
text_temp = embed_text(text_temp)
pridict = model.predict(np.expand_dims(text_temp,0))
print(pridict)
def predict(text):
    input_ids = embed_text(text)
    pridictions = model.predict(np.expand_dims([input_ids], 0))
    return {pridictions}
@app.post('/predict')
def predict_call(text : str = Body(...)):  # put application's code here
    text = embed_text(text)
    pridict = model.predict(np.expand_dims(text, 0))
    if (pridict == 0):
        output = "clean"
    elif (pridict == 1):
        output = "offensive"
    else:
        output = "toxic"
    return output

@app.get('/predict')
def say_Hello():
    text_temp = "Say Hello"
    text_temp = embed_text(text_temp)
    pridict = model.predict(np.expand_dims(text_temp, 0))
    output = ""
    if (pridict == 0):
        output = "clean"
    elif(pridict == 1):
        output = "offensive"
    else :
        output = "toxic"
    return output
