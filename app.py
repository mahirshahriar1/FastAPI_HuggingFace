from fastapi import FastAPI
from fastapi.responses import JSONResponse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_name = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post('/predict_emotion/')
async def predict_emotion(input_text: str):
    inputs = tokenizer(input_text, return_tensors='pt')
    print(input_text)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    y_hat_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][y_hat_class].item()
    label = model.config.id2label[y_hat_class]

    output={'prediction': label, 'confidence' : confidence}

    return JSONResponse(output)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
