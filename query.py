import requests

text = "I am sad"

url = f'http://127.0.0.1:8000/predict_emotion/?input_text={text}'

headers = {'accept': 'application/json'}

response = requests.post(url, headers=headers)

print(response.json())