import base64
import json
import requests
from typing import Dict, Any, Optional
from PIL import Image
import io
from app.core import config


class SimpleWatsonxClient:
    """Simplified Watsonx client using direct API calls instead of the SDK"""

    def __init__(self):
        self.api_key = config.WATSONX_API_KEY
        self.project_id = config.WATSONX_PROJECT_ID
        self.url = config.WATSONX_URL
        self.access_token = None

    def get_access_token(self) -> str:
        """Get access token for Watsonx API"""
        if self.access_token:
            return self.access_token

        auth_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key,
        }

        response = requests.post(auth_url, headers=headers, data=data)
        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_product_description(self, image_path: str) -> Dict[str, Any]:
        """Generate product description from image using Watsonx API"""

        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode the image
            image_base64 = self.encode_image(image_path)

            # Your specific prompt
            prompt = """You are a detail-focused visual inspector.

Goal: From ONE photo of a single item, produce a COMPLETE structured record. Favor **best-guess, high-coverage outputs**. Use visual cues (shape, components, texture), printed text, icons, packaging hints, and common-sense priors. If something is uncertain, still provide the **most probable** value and reflect that in the confidence score. Use "unknown" **only** when there is truly no reasonable inference.

For color and material analysis, identify both PRIMARY and SECONDARY characteristics when present. For example, if an item is mostly white with blue accents, the primary color is "white" and secondary could be "blue" with an estimated percentage. Only include secondary characteristics if they are genuinely visible and significant (typically 15%+ of the item).

Return **ONLY** a valid JSON object with this exact schema:

{
  "fields": {
    "condition": "new|open_box|used|damaged|unknown",
    "colour": "black|white|gray|silver|gold|red|blue|green|brown|beige|transparent|unknown",
    "colour_secondary": {
      "color": "black|white|gray|silver|gold|red|blue|green|brown|beige|transparent|null",
      "percentage": 0
    },
    "material": "plastic|metal|textile|paper|cardboard|wood|glass|ceramic|other|unknown",
    "material_secondary": {
      "material": "plastic|metal|textile|paper|cardboard|wood|glass|ceramic|other|null",
      "percentage": 0
    },
    "category": "apparel|electronics|housewares|toys|tools|books|beauty|sports|other|unknown",
    "weight": { "value": null, "unit": "g|kg|lb|oz|null" },
    "print_label": true,
    "sort": "known_overgoods|vague_overgoods",
    "additional_info": ""
  },
  "confidence": {
    "condition": 0.0,
    "colour": 0.0,
    "colour_secondary": 0.0,
    "material": 0.0,
    "material_secondary": 0.0,
    "category": 0.0,
    "weight": 0.0,
    "print_label": 0.0,
    "sort": 0.0
  },
  "evidence": {
    "ocr_like_text": "",
    "visible_marks": "",
    "image_refs": ["img"]
  },
  "needs_review": false,
  "description": ""
}

Instructions & heuristics:
- **Maximize filled fields.** Choose the most likely option; adjust the corresponding confidence in [0,1].
- **Condition** describes the item itself (not just the cardboard). Packaging wear can imply open_box.
- **Colour**: pick the dominant visible surface colour. If two are equally dominant, choose the one that visually covers more of the item; note the secondary in `additional_info`.
- **Material**: infer from texture/finish (e.g., matte polymer, brushed metal, woven fabric); use "other" rather than "unknown" when a reasonable class exists.
- **Category**: prefer the closest fit from the list; if borderline, choose the most probable and reflect uncertainty via confidence.
- **Weight**: ONLY if printed/legible on the image; otherwise leave value=null and unit=null but still include a brief rationale in `additional_info` if an apparent size/form suggests a typical range (do NOT invent numbers).
- **print_label = true** if a barcode/QR/SKU/address block or shipping label is visibly present (even if partially unreadable).
- **sort** = "known_overgoods" if there is any strong identifier (barcode/SKU/model/no. or clearly addressed label); otherwise "vague_overgoods".
- **additional_info**: one short sentence with key visual cues or markings that informed your choices.
- **description**: 1 concise sentence summarizing the item using ONLY information implied by the image (no brands unless clearly printed).

Output JSON only—no extra prose outside the JSON."""

            # Prepare the API request headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Watsonx API payload for chat with vision
            payload = {
                "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                "project_id": self.project_id,
            }

            # Make the API call using the chat endpoint
            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"Making API call to: {api_url}")
            print(f"Project ID: {self.project_id}")

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            print(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                # For chat API, the response format is different
                if "choices" in result:
                    # Chat API format
                    generated_text = result["choices"][0]["message"]["content"]
                else:
                    # Fallback to text generation format
                    generated_text = result.get("results", [{}])[0].get(
                        "generated_text", ""
                    )

                print(f"Generated text: {generated_text[:200]}...")

                # Try to parse JSON from the response
                try:
                    # Find JSON in the response
                    start_idx = generated_text.find("{")
                    end_idx = generated_text.rfind("}") + 1

                    if start_idx != -1 and end_idx != -1:
                        json_str = generated_text[start_idx:end_idx]
                        parsed_result = json.loads(json_str)

                        return {
                            "success": True,
                            "data": parsed_result,
                            "raw_response": generated_text,
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Could not find JSON in response",
                            "raw_response": generated_text,
                        }

                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"JSON parsing error: {str(e)}",
                        "raw_response": generated_text,
                    }
            else:
                error_msg = f"API call failed: {response.status_code} - {response.text}"
                print(error_msg)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def generate_search_embedding(self, image_path: str) -> Dict[str, Any]:
        """Generate detailed description for search purposes using AI"""

        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode the image
            image_base64 = self.encode_image(image_path)

            # Search-optimized prompt
            search_prompt = """You are an expert at analyzing images for search and matching purposes.

Analyze this image and provide a detailed, specific description that would help match it against similar items. Focus on:

1. **Primary item identification**: What is the main item in the image?
2. **Key visual features**: Shape, size, color, distinctive markings
3. **Brand/model information**: Any visible text, logos, model numbers
4. **Condition and packaging**: How is it packaged, what condition does it appear to be in?
5. **Category and type**: Specific category (electronics, clothing, etc.) and subcategory

Be very specific and detailed. Mention any text, numbers, or distinctive features you can see. This description will be used to find similar items.

Provide a comprehensive description in 2-3 sentences that captures all the key identifying features."""

            # Prepare the API request headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Watsonx API payload for search description
            payload = {
                "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": search_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                "project_id": self.project_id,
            }

            # Make the API call
            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"Making search embedding API call...")

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()

                # For chat API, the response format is different
                if "choices" in result:
                    # Chat API format
                    description = result["choices"][0]["message"]["content"]
                else:
                    # Fallback to text generation format
                    description = result.get("results", [{}])[0].get(
                        "generated_text", ""
                    )

                print(f"Search description generated: {description[:100]}...")

                return {"success": True, "description": description.strip()}
            else:
                print(f"Search embedding API failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API call failed: {response.status_code}",
                }

        except Exception as e:
            print(f"Search embedding error: {e}")
            return {"success": False, "error": f"Embedding generation error: {str(e)}"}

    def process_natural_language_search(self, query: str) -> Dict[str, Any]:
        """Process natural language search query and convert to search terms"""

        try:
            # Get access token
            access_token = self.get_access_token()

            # Natural language processing prompt
            nl_prompt = f"""You are a search query processor for an overgoods inventory system.

The user has entered this search query: "{query}"

Convert this natural language query into specific search terms that would help find matching items in an inventory. Consider:

1. **Item type/category**: What kind of item are they looking for?
2. **Brand/model**: Any specific brands or models mentioned?
3. **Color**: Any colors mentioned?
4. **Material**: Any materials mentioned?
5. **Condition**: Any condition indicators?
6. **Features**: Any specific features or characteristics?

Return a focused search description that captures the key elements they're looking for. Be specific and use terms that would match against item descriptions.

Provide a concise search description (1-2 sentences) that captures the essential search criteria."""

            # Prepare the API request headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Watsonx API payload for natural language processing
            payload = {
                "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": nl_prompt}],
                    }
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                "project_id": self.project_id,
            }

            # Make the API call
            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"Processing natural language query: {query}")

            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                # For chat API, the response format is different
                if "choices" in result:
                    # Chat API format
                    processed_query = result["choices"][0]["message"]["content"]
                else:
                    # Fallback to text generation format
                    processed_query = result.get("results", [{}])[0].get(
                        "generated_text", ""
                    )

                print(f"Processed query: {processed_query[:100]}...")

                return {"success": True, "processed_query": processed_query.strip()}
            else:
                print(f"Natural language processing failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API call failed: {response.status_code}",
                }

        except Exception as e:
            print(f"Natural language processing error: {e}")
            return {"success": False, "error": f"Processing error: {str(e)}"}
