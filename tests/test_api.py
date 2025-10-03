#!/usr/bin/env python3
"""
Test script for the Overgoods AI System API
"""

import requests
import json
from PIL import Image, ImageDraw
import os


# Create a simple test image
def create_test_image():
    """Create a simple test image for demonstration"""
    img = Image.new("RGB", (300, 200), color="red")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "TEST ITEM", fill="white")
    draw.rectangle([50, 100, 250, 150], outline="white", width=3)

    os.makedirs("test_images", exist_ok=True)
    img.save("test_images/test_item.jpg")
    return "test_images/test_item.jpg"


def test_search_api():
    """Test the search API with form data"""
    print("🔍 Testing Search API...")

    # Test search by keywords
    response = requests.post(
        "http://localhost:8000/api/search", data={"keywords": "leather", "n_results": 3}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Search successful! Found {result['count']} results")
        for i, item in enumerate(result["results"][:2], 1):
            print(
                f"  {i}. {item['metadata']['title']} - {item['metadata']['color']} {item['metadata']['material']}"
            )
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(response.text)


def test_description_api():
    """Test the description generation API"""
    print("\n📝 Testing Description API...")

    # Create test image
    test_image_path = create_test_image()

    # Test description generation
    with open(test_image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(
            "http://localhost:8000/api/generate-description", files=files
        )

    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print("✅ Description generation successful!")
            data = result["data"]
            print(f"  Title: {data['title']}")
            print(f"  Description: {data['description']}")
            print(f"  Condition: {data['condition']}")
            print(f"  Confidence: {data['confidence']}/10")
        else:
            print(f"❌ Description generation failed: {result['error']}")
    else:
        print(f"❌ API call failed: {response.status_code}")
        print(response.text)


def test_image_search():
    """Test image-based search"""
    print("\n🖼️ Testing Image Search...")

    test_image_path = create_test_image()

    with open(test_image_path, "rb") as f:
        files = {"file": f}
        data = {"n_results": 3}
        response = requests.post(
            "http://localhost:8000/api/search", files=files, data=data
        )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Image search successful! Found {result['count']} results")
        for i, item in enumerate(result["results"][:2], 1):
            similarity = 1 - item.get("distance", 0)
            print(f"  {i}. {item['metadata']['title']} - Similarity: {similarity:.2f}")
    else:
        print(f"❌ Image search failed: {response.status_code}")
        print(response.text)


def main():
    """Run all tests"""
    print("🏢 Testing Overgoods AI System API")
    print("=" * 50)

    try:
        # Test if server is running
        response = requests.get("http://localhost:8000/")
        if response.status_code != 200:
            print("❌ Server is not running! Please start it with: python main.py")
            return

        print("✅ Server is running!")

        # Run tests
        test_search_api()
        test_description_api()
        test_image_search()

        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("\n📖 To use the web interface, open: http://localhost:8000")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server! Please start it with: python main.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    main()
