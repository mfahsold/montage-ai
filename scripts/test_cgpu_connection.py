#!/usr/bin/env python3
"""
Test script for cgpu/Gemini connection.
Verifies that the cgpu serve endpoint is reachable and responding.
"""

import os
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CGPU_HOST = os.environ.get("CGPU_HOST", "localhost")
CGPU_PORT = os.environ.get("CGPU_PORT", "8090")
CGPU_MODEL = os.environ.get("CGPU_MODEL", "gemini-2.0-flash")
BASE_URL = f"http://{CGPU_HOST}:{CGPU_PORT}/v1"

def test_connection():
    print(f"Testing connection to cgpu at {BASE_URL}...")
    
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key="unused", # cgpu ignores API key, auth is handled by the server process
            timeout=30
        )

        # Test the 'responses' endpoint which cgpu exposes
        if hasattr(client, 'responses'):
            logger.info("Client has 'responses' attribute (correct for cgpu).")
            logger.info(f"Sending test request to model: {CGPU_MODEL}...")
            
            response = client.responses.create(
                model=CGPU_MODEL,
                instructions="You are a helpful assistant.",
                input="Hello! Just checking the connection."
            )
            
            print("\n✅ Connection successful!")
            print(f"Response: {response.output_text if hasattr(response, 'output_text') else response}")
        else:
            logger.warning("Client does NOT have 'responses' attribute. This might not be the cgpu endpoint.")

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is 'cgpu serve' running? (Check with: ps aux | grep cgpu)")
        print("2. Is the port correct? (Default: 8090)")
        print("3. Are GOOGLE_API_KEY or GEMINI_API_KEY set in the environment where cgpu is running?")

if __name__ == "__main__":
    test_connection()
