#!/usr/bin/env python3
"""
Test Watsonx API connection
"""

import requests
import base64
import json
from simple_watsonx_client import SimpleWatsonxClient
import config


def test_watsonx_connection():
    """Test basic Watsonx connection"""
    print("🧪 Testing Watsonx API Connection...")
    print(f"API Key: {config.WATSONX_API_KEY[:10]}...")
    print(f"Project ID: {config.WATSONX_PROJECT_ID}")
    print(f"URL: {config.WATSONX_URL}")

    client = SimpleWatsonxClient()

    # Test getting access token
    try:
        token = client.get_access_token()
        print(f"✅ Access token obtained: {token[:20]}...")
    except Exception as e:
        print(f"❌ Failed to get access token: {e}")
        return

    # Test simple text generation (no image)
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
            "input": "Hello, can you respond with just the word 'SUCCESS'?",
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 10,
                "temperature": 0.1,
            },
            "project_id": client.project_id,
        }

        api_url = f"{client.url}/ml/v1/text/generation?version=2023-05-29"
        print(f"Making test API call to: {api_url}")

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("✅ Basic API connection working!")
        else:
            print(f"❌ API call failed: {response.status_code}")

    except Exception as e:
        print(f"❌ API test failed: {e}")


if __name__ == "__main__":
    test_watsonx_connection()
