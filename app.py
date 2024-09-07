from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pydantic import BaseModel
from text_to_speech import convert_text_to_speech

app = FastAPI()

@app.post("/")
async def post():
    return {"message": "hello from the post route"}

@app.get("/")
async def root():
    return {"message": "hello world"}

# Define the endpoint for the text_to_speech API
@app.post('/text_to_speech/')
async def create_speech(text: str, background_tasks: BackgroundTasks):
    """
    API endpoint to convert text to speech and return a success message.
    The actual audio file is not directly returned due to security and scalability concerns.
    """
    try:
        # Validate input length (optional)
        if len(text) > 600:
            raise HTTPException(status_code=400, detail="Text too long. Please limit to 550 characters or less.")

        # Run conversion in background
        filename = background_tasks.add_task(convert_text_to_speech, text)             
        
        return JSONResponse({"message": f"Text-to-speech conversion initiated. Audio file saved as: {filename}"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    

# Define the endpoint for the predict_emotion API

model_name = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

## This takes parameters from the URL
# @app.post('/predict_emotion/')
# async def predict_emotion(input_text: str):
#     """
#         API endpoint to predict the emotion of a given text.
#     """
#     inputs = tokenizer(input_text, return_tensors='pt')
#     print(input_text)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1)
#     y_hat_class = torch.argmax(probabilities, dim=1).item()

#     confidence = probabilities[0][y_hat_class].item()
#     label = model.config.id2label[y_hat_class]

#     output={'prediction': label, 'confidence' : confidence}

#     return JSONResponse(output)



## This takes parameters from the request body

# Define a Pydantic model to expect input_text from the request body
# Expect the input_text as part of the body
class TextInput(BaseModel):
    input_text: str 

@app.post('/predict_emotion/')
async def predict_emotion(text: TextInput):
    """
    API endpoint to predict the emotion of a given
    text. This version expects the input text in the
    request body.
    """
    # Access the input text from the request body
    input_text = text.input_text
    print(input_text)

    # Simulate prediction response
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    y_hat_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][y_hat_class].item()

    label = model.config.id2label[y_hat_class]

    output = {'prediction': label, 'confidence': confidence}
    return JSONResponse(output)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
