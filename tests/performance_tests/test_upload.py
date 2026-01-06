
import requests
import os

# Create a dummy file
with open('test_upload.txt', 'wb') as f:
    f.write(b'x' * 1024 * 1024) # 1MB

url = 'http://localhost:5001/api/upload'
files = {'file': open('test_upload.txt', 'rb')}
data = {'type': 'video'}

try:
    print(f"Uploading to {url}...")
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
finally:
    os.remove('test_upload.txt')
