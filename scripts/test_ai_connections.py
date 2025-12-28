#!/usr/bin/env python3
"""
Test script to verify AI backend connectivity.
Run this from the host machine (not inside Docker).
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_status(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}: {details}")

def test_cgpu():
    """Test connection to cgpu serve (Gemini)."""
    print("\nTesting cgpu (Gemini)...")
    
    # cgpu serve runs on localhost on the host machine
    port = os.environ.get("CGPU_PORT", "8090")
    url = f"http://127.0.0.1:{port}/v1/models"
    
    try:
        # Skip listing models as cgpu doesn't support it well and it might cause state issues
        pass

        # Always try generation, even if listing fails
        # cgpu uses the /v1/responses endpoint (OpenAI Responses API)
        # We use curl via subprocess because requests seems to have issues with cgpu's server implementation
        import subprocess
        
        chat_url = f"http://127.0.0.1:{port}/v1/responses"
        model = os.environ.get("CGPU_MODEL", "gemini-flash-latest")
        
        curl_cmd = [
            "curl", "-s", "-X", "POST", chat_url,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"model": model, "input": "Hello"})
        ]
        
        try:
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                try:
                    response_json = json.loads(result.stdout)
                    if 'output_text' in response_json:
                        content = response_json['output_text']
                        print_status("cgpu generation", True, f"Response: {content}")
                    elif 'error' in response_json and response_json['error']:
                         print_status("cgpu generation", False, f"API Error: {response_json['error']}")
                    else:
                        # Fallback for other formats or errors
                        print_status("cgpu generation", True, f"Response: {str(response_json)[:100]}...")
                except json.JSONDecodeError:
                     print_status("cgpu generation", False, f"Invalid JSON: {result.stdout}")
            else:
                print_status("cgpu generation", False, f"Curl failed: {result.stderr}")
                
        except Exception as e:
                print_status("cgpu generation", False, f"Error: {e}")

    except requests.exceptions.ConnectionError:
        print_status("cgpu serve", False, "Connection refused. Is 'cgpu serve' running?")
    except Exception as e:
        print_status("cgpu serve", False, str(e))

def test_google_ai_direct():
    """Test direct connection to Google AI API."""
    print("\nTesting Google AI (Direct)...")
    
    # Force reload of env vars
    os.environ.pop("GOOGLE_AI_MODEL", None)
    load_dotenv(override=True)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print_status("Google AI", False, "No API Key found (GEMINI_API_KEY or GOOGLE_API_KEY)")
        return

    model = os.environ.get("GOOGLE_AI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Hello, are you working?"}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if response.status_code == 200:
            print_status("Google AI", True, "Connection successful")
            try:
                text = response.json()['candidates'][0]['content']['parts'][0]['text']
                print(f"   Response: {text.strip()}")
            except:
                print(f"   Response structure unexpected: {response.text[:100]}...")
        else:
            print_status("Google AI", False, f"Status {response.status_code}: {response.text}")
    except Exception as e:
        print_status("Google AI", False, str(e))

def test_ollama():
    """Test connection to local Ollama."""
    print("\nTesting Ollama (Local)...")
    
    # Ollama usually runs on 11434
    url = "http://127.0.0.1:11434/api/tags"
    
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print_status("Ollama", True, "Connection successful")
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"   Available models: {', '.join(models)}")
        else:
            print_status("Ollama", False, f"Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_status("Ollama", False, "Connection refused (not running)")
    except Exception as e:
        print_status("Ollama", False, str(e))

if __name__ == "__main__":
    print("=== AI Backend Connectivity Test ===")
    test_cgpu()
    test_google_ai_direct()
    test_ollama()
    print("\nDone.")
