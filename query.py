# import requests
#
# text = "I am sad"
#
# url = f'http://127.0.0.1:8000/predict_emotion/?input_text={text}'
#
# headers = {'accept': 'application/json'}
#
# response = requests.post(url, headers=headers)
#
# print(response.json())

import requests

text = "Hi my name is mahir shahriar tamim"

url = f'http://127.0.0.1:8000/text_to_speech/?text={text}'
headers = {'Content-Type': 'application/json'}
data = {'text': text}  # Include the required 'text' parameter

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print(f"Text-to-speech conversion initiated. Message: {response.json()['message']}")
else:
    print(f"Error: {response.status_code} - {response.text}")