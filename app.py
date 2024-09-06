import time

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import asyncio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import nltk
from nltk.tokenize import word_tokenize


def convert_text_to_speech(text):
    """Performs text-to-speech conversion using the specified model."""
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


    inputs = processor(text=text, return_tensors="pt")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)


    # Save the generated audio to a temporary file
    filename = f"speech_{round(time.time())}.wav"  # Generate unique filename
    sf.write(filename, speech.numpy(), samplerate=16000)

    return filename


app = FastAPI()

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
