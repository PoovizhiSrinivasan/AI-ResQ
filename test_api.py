import requests
import os

url = "http://127.0.0.1:5000/predict"

audio_path = "distress_1423.wav"   # change if needed

# Check if file exists
if not os.path.exists(audio_path):
    print("❌ Audio file not found:", audio_path)
    exit()

with open(audio_path, "rb") as f:

    files = {
        "audio": f
    }

    response = requests.post(url, files=files)

print("Status Code:", response.status_code)

try:
    print("Response:", response.json())
except:
    print("Server returned non-JSON response:", response.text)