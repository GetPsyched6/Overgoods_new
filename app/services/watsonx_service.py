import base64
import json
import re
import requests
from typing import Dict, Any, Optional, List
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

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Get access token for Watsonx API"""
        if self.access_token and not force_refresh:
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

    def get_image_mime_type(self, image_path: str) -> str:
        """Detect image MIME type from file extension"""
        import os

        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_types.get(ext, "image/jpeg")

    def _parse_multiple_objects_fallback(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse markdown/text formatted multiple objects response as fallback"""
        try:
            text_lower = text.lower()

            # Initialize defaults
            multiple_objects = False
            count = 1
            brief_description = "Item"

            # Try to extract count
            # Look for patterns like "1", "single", "one object", "count: 1"
            count_patterns = [
                r"count[:\s*]+(\d+)",
                r"(\d+)\s+(?:distinct|main|object|item)",
                r"number[^:]*:[^\d]*(\d+)",
            ]

            for pattern in count_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    count = int(match.group(1))
                    break

            # Determine if multiple objects
            if count > 1:
                multiple_objects = True
            elif any(
                word in text_lower
                for word in [
                    "single",
                    "one object",
                    "1 object",
                    "only 1",
                    "count: 1",
                    "count:** 1",
                ]
            ):
                multiple_objects = False
                count = 1
            elif any(
                word in text_lower for word in ["multiple", "several", "more than one"]
            ):
                multiple_objects = True
                if count == 1:  # If we didn't find count yet
                    count = 2

            # Try to extract brief description
            desc_patterns = [
                r"brief[_ ]description[:\s*\*]+([^\n\*]+)",
                r"description[:\s*\*]+([^\n\*]+)",
                r"main item[s]?[:\s*\*]+([^\n\*]+)",
                r"item[:\s*\*]+([^\n\*]+)",
            ]

            for pattern in desc_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    brief_description = match.group(1).strip().rstrip("*").strip()
                    # Capitalize first letter
                    brief_description = (
                        brief_description[0].upper() + brief_description[1:]
                        if brief_description
                        else "Item"
                    )
                    break

            # If still no description, look for recognizable product terms
            if brief_description == "Item":
                product_terms = [
                    "xbox",
                    "controller",
                    "laptop",
                    "mouse",
                    "keyboard",
                    "cable",
                    "plate",
                    "game",
                    "console",
                    "monitor",
                    "phone",
                    "tablet",
                ]
                for term in product_terms:
                    if term in text_lower:
                        brief_description = term.capitalize()
                        break

            return {
                "multiple_objects": multiple_objects,
                "count": count,
                "brief_description": brief_description,
            }
        except Exception as e:
            print(f"Fallback parsing error: {e}")
            return None

    def check_multiple_objects(self, image_path: str) -> Dict[str, Any]:
        """Check if image contains multiple distinct objects"""

        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode the image
            image_base64 = self.encode_image(image_path)

            # Multiple object detection prompt
            prompt = """CRITICAL: Your response must be PURE JSON. Start with { and end with }. No markdown, no labels, no text before or after.

Analyze this image and determine if there are multiple DISTINCT MAIN items present.

IMPORTANT RULES:
1. IGNORE background items (tables, mats, walls, shelves, etc.)
2. IGNORE packaging materials (boxes, bubble wrap, bags) - focus on what's INSIDE
3. We're looking for DIFFERENT types of MAIN items, not multiple of the same thing
4. Only count items that appear to be the PRIMARY subject of the photo

Examples:
- Xbox controller in a box on a mat = SINGLE object (mat is background)
- Box with 5 identical plates = SINGLE object (multiple of same item)
- Laptop AND mouse both packaged together = MULTIPLE objects (different items)
- 3 books of the same type = SINGLE object (multiple of same)
- Shirt AND shoes in same package = MULTIPLE objects (different items)

CRITICAL OUTPUT REQUIREMENTS:
- Your response must START with { and END with }
- Do NOT add markdown (no ```, no ```json)
- Do NOT add labels (no "Answer:", no "JSON Output:", no "Here is")
- Do NOT add explanatory text before or after the JSON
- Just output the raw JSON object immediately

Schema:
{
  "multiple_objects": true or false,
  "count": number of distinct main object types,
  "brief_description": "brief description of main item(s)"
}

Begin your response with { now:
{"""

            # Prepare the API request headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Watsonx API payload
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
                    "max_new_tokens": 100,
                    "temperature": 0.0,
                    "top_p": 0.9,
                },
                "project_id": self.project_id,
            }

            # Make the API call
            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"Checking for multiple objects...")

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            # If 401, refresh token and retry once
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

            if response.status_code == 200:
                result = response.json()

                # For chat API, the response format is different
                if "choices" in result:
                    generated_text = result["choices"][0]["message"]["content"]
                else:
                    generated_text = result.get("results", [{}])[0].get(
                        "generated_text", ""
                    )

                print(f"Multiple object check response: {generated_text[:200]}...")

                # Try to parse JSON from the response
                try:
                    # Strip markdown code blocks if present
                    json_str = generated_text.strip()

                    # Handle markdown code blocks: ```json\n{...}\n``` or ```\n{...}\n```
                    if "```" in json_str:
                        code_block_match = re.search(
                            r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL
                        )
                        if code_block_match:
                            json_str = code_block_match.group(1).strip()
                            print(f"Extracted JSON from markdown code block")

                    # If it starts with text before JSON, extract just the JSON part
                    if not json_str.startswith("{"):
                        start_idx = json_str.find("{")
                        if start_idx != -1:
                            json_str = json_str[start_idx:]

                    # If it has text after JSON, extract just the JSON part
                    if not json_str.endswith("}"):
                        end_idx = json_str.rfind("}") + 1
                        if end_idx > 0:
                            json_str = json_str[:end_idx]

                    # CRITICAL: Unescape JSON BEFORE checking if it's valid
                    if (
                        "\\{" in json_str
                        or "\\}" in json_str
                        or "\\[" in json_str
                        or "\\]" in json_str
                    ):
                        print("Detected escaped JSON, unescaping...")
                        json_str = json_str.replace("\\{", "{").replace("\\}", "}")
                        json_str = json_str.replace("\\[", "[").replace("\\]", "]")
                        json_str = json_str.replace('\\"', '"')

                    if json_str.startswith("{") and json_str.endswith("}"):
                        parsed_result = json.loads(json_str)

                        return {
                            "success": True,
                            "data": parsed_result,
                            "raw_response": generated_text,
                        }
                    else:
                        # FALLBACK: Parse markdown/text format
                        print("No JSON found, attempting text parsing...")
                        fallback_result = self._parse_multiple_objects_fallback(
                            generated_text
                        )
                        if fallback_result:
                            print("Fallback parsing successful!")
                            return {
                                "success": True,
                                "data": fallback_result,
                                "raw_response": generated_text,
                                "parsing_warning": "AI returned non-JSON format for multiple objects check. Fallback parsing was used.",
                            }
                        return {
                            "success": False,
                            "error": "Could not find valid JSON in response",
                            "raw_response": generated_text,
                        }

                except json.JSONDecodeError as e:
                    # FALLBACK: Parse markdown/text format
                    print(f"JSON parsing failed: {e}. Attempting text parsing...")
                    fallback_result = self._parse_multiple_objects_fallback(
                        generated_text
                    )
                    if fallback_result:
                        print("Fallback parsing successful!")
                        return {
                            "success": True,
                            "data": fallback_result,
                            "raw_response": generated_text,
                            "parsing_warning": "AI returned non-JSON format for multiple objects check. Fallback parsing was used.",
                        }
                    return {
                        "success": False,
                        "error": f"JSON parsing error: {str(e)}",
                        "raw_response": generated_text,
                    }
            else:
                error_msg = f"API call failed: {response.status_code}"
                print(error_msg)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Error checking multiple objects: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def generate_product_description(
        self, image_path: str, multiple_objects_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate product description from image using Watsonx API"""

        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode the image
            image_base64 = self.encode_image(image_path)

            # Build prompt based on whether we have multiple objects
            if multiple_objects_data and multiple_objects_data.get("multiple_objects"):
                object_count = multiple_objects_data.get("count", 2)
                # Your specific prompt for MULTIPLE objects
                prompt = f"""CRITICAL FORMAT RULE: Your response must be PURE JSON starting with {{ and ending with }}. No markdown code blocks, no "Answer:", no "JSON Output:", no text before or after the JSON. Output ONLY the JSON object.

You are a detail-focused visual inspector with expertise in product identification.

**CRITICAL: MULTIPLE OBJECTS DETECTED - ANALYZE EACH SEPARATELY**
This image contains {object_count} distinct objects. You must analyze EACH object individually and return data as ARRAYS.

For each field, provide an array with {object_count} entries, one for each object in order (left to right, or primary to secondary).

 If something is uncertain, still provide the most probable value and reflect that in the confidence score. Be harsh with the confidence scores, if you're really not sure about something, let it be low. Use "unknown" only when there is truly no reasonable inference.

**CRITICAL: IDENTIFY BRANDS AND MODELS FOR EACH OBJECT**
- Look for ANY visible branding, logos, or model identifiers on EACH object
- Each object gets its own brand, model, color, material, etc.
- Examples: Game Boy + Game Cartridge → ["Nintendo", "Nintendo"], ["Game Boy Advance SP", "Super Mario Advance"]

Schema (arrays must have EXACTLY {object_count} entries):

{{
  "fields": {{
    "condition": ["new|open_box|used|damaged|unknown", ...],
    "colour": ["black|white|gray|silver|gold|red|blue|green|brown|beige|transparent|unknown", ...],
    "material": ["plastic|metal|chemical|fabric|fibreglass|fur|liquid|perishable|rubber|stone|wood|textile|paper|cardboard|glass|ceramic|other|unknown", ...],
    "category": ["automotive/vehicle|baby goods|building supplies/materials|chemicals|clothing/shoes/accessories|collectibles|computers/networking|consumer electronics|drugs/pharmaceuticals|electrical/lighting|entertainment media|fabric|food/beverages|gift cards|health/beauty|housewares|toys|tools|books|sports|other|unknown", ...],
    "brand": ["", ...],
    "model": ["", ...],
    "quantity": [0, ...],
    "upc": ["", ...],
    "additional_info": ["", ...]
  }},
  "confidence": {{
    "condition": [0.0, ...],
    "colour": [0.0, ...],
    "material": [0.0, ...],
    "category": [0.0, ...],
    "brand": [0.0, ...],
    "model": [0.0, ...],
    "quantity": [0.0, ...],
    "upc": [0.0, ...]
  }},
  "descriptions": ["Rich detailed description of object 1 with brand, model, condition, distinctive features", "Rich detailed description of object 2...", ...],
  "global": {{
    "object_count": {object_count},
    "print_label": false,
    "ocr_text": "ALL visible text from the image",
    "visible_marks": "logos, serial numbers, any identifiers"
  }}
}}

Instructions:
- Each array must have EXACTLY {object_count} entries
- Order: left to right, or primary object first
- **descriptions**: Write 2-3 sentence descriptions of what you SEE for each object - physical appearance, visible text/logos, condition, colors, materials. DO NOT invent backstory, purpose, or content. DO NOT describe what the item does unless explicitly printed on it. Stick to observable physical details only.
- **quantity**: For each object type, COUNT the ACTUAL NUMBER of individual items visible. If you see 10 plates (even if called "1 set"), quantity=10. If you see 1 Game Boy, quantity=1. Count individual items, NOT sets. Be precise and count carefully!
- **upc**: For each object, ONLY extract if there is a visible UPC/EAN/ISBN product barcode on that specific item or its packaging (the vertical black and white stripes with 12-13 digit numbers underneath). Extract the digits ONLY if you can clearly read them below the barcode stripes. This field is EXCLUSIVELY for barcode numbers - do NOT put these numbers in the model field. If no barcode is visible for that object, use empty string "". Do NOT fill this with license numbers, serial numbers, or other non-barcode text.
- **additional_info**: Include EVERYTHING - model numbers, serial numbers, generation info, special editions, any visible text on that specific object, unique identifiers, version details, region codes, anything that helps identify the exact variant
- **Brand/Model**: Be aggressive in identification - use button layouts, port configs, design language, any visual cues. Model should be the MOST SPECIFIC product model/generation/variant (e.g., "Xbox Wireless Controller - Series X|S", "Game Boy Advance SP", "PlayStation 5 - Disc Edition"). Include generation/version info if visible (like "Series X|S", "Gen 2", "Pro"). Model is the product's MODEL NAME, NOT a UPC/EAN/ISBN barcode number. If no specific model/version visible, use generic name. If truly unsure, leave empty "".
- **global.object_count**: MUST be {object_count} (the number of distinct object TYPES, not total quantity)
- **global.print_label**: true ONLY if there's a shipping label, barcode sticker, or SKU label visible (NOT product branding)
- **global.ocr_text**: Extract ALL readable text from the entire image including product names, model numbers, any text on objects or packaging (limit to most important text if too long)
- **global.visible_marks**: Note any logos, serial numbers, QR codes, barcodes visible anywhere in the image

CRITICAL OUTPUT REQUIREMENTS:
- Your response MUST START with {{ and END with }}
- Do NOT add markdown formatting (no ```, no ```json, no **bold**)
- Do NOT add text labels (no "Answer:", no "JSON Output:", no "Here is the JSON:")
- Do NOT add explanatory text before or after the JSON
- Output ONLY the raw JSON object - nothing else

Begin your response with {{ now:
{{"""
            else:
                # Single object prompt (original)
                prompt = """CRITICAL FORMAT RULE: Your response must be PURE JSON starting with { and ending with }. No markdown code blocks, no "Answer:", no "JSON Output:", no text before or after the JSON. Output ONLY the JSON object.

You are a detail-focused visual inspector with expertise in product identification.

Goal: From ONE photo of a single item, produce a COMPLETE structured record with MAXIMUM detail extraction. Favor best-guess, high-coverage outputs. Use visual cues (shape, components, texture, design language), printed text, logos, icons, packaging hints, model-specific features, and common-sense priors. If something is uncertain, still provide the most probable value and reflect that in the confidence score. Be harsh with the confidence scores, if you're really not sure about something, let it be low. Use "unknown" only when there is truly no reasonable inference.

**CRITICAL: IDENTIFY BRANDS AND MODELS**
- Look for ANY visible branding, logos, or model identifiers
- Use design language, button layouts, port configurations, and distinctive features to infer brands/models
- Examples: Xbox controller → identify if Xbox 360/One/Series X/S based on button layout, D-pad design, home button style
- Include brand and model in description and additional_info when identifiable

For color and material analysis, identify both PRIMARY and SECONDARY characteristics when present. For example, if an item is mostly white with blue accents, the primary color is "white" and secondary could be "blue" with an estimated percentage. Only include secondary characteristics if they are genuinely visible and significant (typically 15%+ of the item).

Schema:

{
  "fields": {
    "condition": "new|open_box|used|damaged|unknown",
    "colour": "black|white|gray|silver|gold|red|blue|green|brown|beige|transparent|unknown",
    "colour_secondary": "black|white|gray|silver|gold|red|blue|green|brown|beige|transparent|null",
    "material": "plastic|metal|chemical|fabric|fibreglass|fur|liquid|perishable|rubber|stone|wood|textile|paper|cardboard|glass|ceramic|other|unknown",
    "material_secondary": "plastic|metal|chemical|fabric|fibreglass|fur|liquid|perishable|rubber|stone|wood|textile|paper|cardboard|glass|ceramic|other|null",
    "category": "automotive/vehicle|baby goods|building supplies/materials|chemicals|clothing/shoes/accessories|collectibles|computers/networking|consumer electronics|drugs/pharmaceuticals|electrical/lighting|entertainment media|fabric|food/beverages|gift cards|health/beauty|housewares|toys|tools|books|sports|other|unknown",
    "brand": "",
    "model": "",
    "quantity": 0,
    "weight": { "value": null, "unit": "g|kg|lb|oz|null" },
    "print_label": true,
    "upc": "",
    "additional_info": ""
  },
  "confidence": {
    "condition": 0.0,
    "colour": 0.0,
    "colour_secondary": 0.0,
    "material": 0.0,
    "material_secondary": 0.0,
    "category": 0.0,
    "brand": 0.0,
    "model": 0.0,
    "quantity": 0.0,
    "weight": 0.0,
    "print_label": 0.0,
    "upc": 0.0
  },
  "evidence": {
    "ocr_like_text": "",
    "visible_marks": "",
    "image_refs": ["img"]
  },
  "description": ""
}

Instructions:
- **Maximize filled fields.** Choose the most likely option; adjust the corresponding confidence in [0,1].
- **Condition**: "new" = unopened/mint; "open_box" = opened but unused; "used" = shows normal wear; "damaged" = broken/defective; "unknown" if unclear.
- **Colour**: pick the dominant visible surface colour. If a significant secondary color exists (15%+ of item), specify it in colour_secondary, otherwise use "null".
- **Material**: infer from texture/finish (e.g., matte polymer, brushed metal, woven fabric); use "other" rather than "unknown" when a reasonable class exists. If a significant secondary material exists (15%+ of item), specify it in material_secondary, otherwise use "null".
- **Category**: prefer the closest fit from the list; if borderline, choose the most probable and reflect uncertainty via confidence.
- **Brand**: Extract from visible logos, text, or infer from distinctive design features. Be aggressive in identification.
- **Model**: Identify the MOST SPECIFIC product model/generation/variant visible (e.g., "Xbox Wireless Controller - Series X|S", "Game Boy Advance SP", "PlayStation 5 - Disc Edition", "iPhone 13 Pro Max"). Include generation/version info if visible on the item or packaging (like "Series X|S", "Gen 2", "Pro", "Plus", etc.). This is the product's MODEL NAME, not a barcode number. CRITICAL: Do NOT put UPC/EAN/ISBN numbers here - those belong ONLY in the upc field. If no specific model/version is visible, use a generic name (e.g., "Xbox Wireless Controller"). If completely unidentifiable, leave empty "".
- **Quantity**: COUNT the ACTUAL NUMBER of individual items visible. If you see 10 plates (even if called "1 set"), quantity=10. If you see 1 controller, quantity=1. Count individual items, NOT sets. Be precise and count carefully!
- **Weight**: ONLY if printed/legible on the image; otherwise leave value=null and unit=null but still include a brief rationale in additional_info if an apparent size/form suggests a typical range (do NOT invent numbers).
- **print_label = true** if a barcode/QR/SKU/address block or shipping label is visibly present (even if partially unreadable).
- **upc**: ONLY for UPC/EAN/ISBN product barcodes (the vertical black and white stripes with 12-13 digit numbers underneath). Extract the digits ONLY if you can clearly read them below the barcode stripes. This field is EXCLUSIVELY for barcode numbers - do NOT put these numbers in the model field. If no barcode is visible or readable, leave this as an empty string "". Do NOT fill this with license numbers, serial numbers, or other non-barcode text.
- **additional_info**: Include ALL identifying details you can extract - model numbers, serial numbers visible, distinctive features, generations, variants, special editions, any text visible on the item or packaging. Be comprehensive.
- **description**: Write 2-3 sentence descriptions of what you SEE in the image - physical appearance, visible text, logos, condition, colors, materials, packaging. DO NOT invent backstory, purpose, or content. DO NOT describe what the item does or contains unless it's explicitly printed on the item. Stick to observable physical details only. Example: "A purple hardcover book titled 'The Queen's Code' by Alison Armstrong" NOT "A book that explores misunderstandings..."

CRITICAL OUTPUT REQUIREMENTS:
- Your response MUST START with { and END with }
- Do NOT add markdown formatting (no ```, no ```json, no asterisks)
- Do NOT add text labels (no "Answer:", no "JSON Output:", no "Response:", no "Here is")
- Do NOT add explanatory text before or after the JSON
- Output ONLY the raw JSON object - nothing else

Begin your response with { now:
{"""

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
                    "temperature": 0.0,  # Most deterministic
                    "top_p": 1.0,  # Not used with greedy, but set to default
                },
                "project_id": self.project_id,
            }

            # Make the API call using the chat endpoint
            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"Making API call to: {api_url}")
            print(f"Project ID: {self.project_id}")

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            print(f"API Response Status: {response.status_code}")

            # If 401, refresh token and retry once
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )
                print(f"Retry API Response Status: {response.status_code}")

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

                print(f"Generated text (first 200 chars): {generated_text[:200]}...")
                print(f"Generated text (last 100 chars): ...{generated_text[-100:]}")
                print(f"Total response length: {len(generated_text)} characters")

                # DEBUG: Print FULL response to see if it's complete
                if len(generated_text) < 2000:  # Only print if reasonably sized
                    print(f"\n===== FULL AI RESPONSE =====")
                    print(generated_text)
                    print(f"===== END OF RESPONSE =====\n")

                # Try to parse JSON from the response
                try:
                    # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
                    json_str = generated_text.strip()
                    print(
                        f"After .strip(): length={len(json_str)}, last char='{json_str[-1]}' (ASCII {ord(json_str[-1])})"
                    )

                    # Track if we used aggressive fallback trimming
                    used_aggressive_trimming = False
                    chars_cut = 0

                    # Handle markdown code blocks: ```json\n{...}\n``` or ```\n{...}\n```
                    if "```" in json_str:
                        # Extract content between code fences
                        # Match ```json or ``` followed by content and closing ```
                        code_block_match = re.search(
                            r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL
                        )
                        if code_block_match:
                            json_str = code_block_match.group(1).strip()
                            print(f"Extracted JSON from markdown code block")

                    # If it starts with text before JSON, extract just the JSON part
                    if not json_str.startswith("{"):
                        start_idx = json_str.find("{")
                        if start_idx != -1:
                            print(
                                f"  Trimming start - found '{{' at position {start_idx}"
                            )
                            json_str = json_str[start_idx:]

                    # If it has text after JSON, extract just the JSON part
                    # We need to find the MATCHING closing brace, not just use rfind()
                    original_len = len(json_str)
                    if not json_str.endswith("}"):
                        last_char = json_str[-1]
                        print(
                            f"  ⚠️  JSON doesn't end with '}}' - last char: '{last_char}' (ASCII: {ord(last_char)})"
                        )
                        print(f"  Last 40 chars: ...{repr(json_str[-40:])}")

                        # Find the matching closing brace by counting nesting levels
                        brace_count = 0
                        end_idx = -1
                        for i, char in enumerate(json_str):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found the matching closing brace for the first opening brace
                                    end_idx = i + 1
                                    break

                        if end_idx > 0:
                            print(
                                f"  ✓ Found MATCHING '}}' at position: {end_idx - 1} (will cut to length {end_idx})"
                            )
                            json_str = json_str[:end_idx]
                            chars_cut = original_len - len(json_str)
                            print(
                                f"  ✂️  CUT OFF {chars_cut} chars! ({original_len} → {len(json_str)})"
                            )
                            print(
                                f"  After trim, last 40 chars: ...{repr(json_str[-40:])}"
                            )
                            # If we cut more than 100 chars, this is aggressive trimming
                            if chars_cut > 100:
                                used_aggressive_trimming = True
                        else:
                            # FALLBACK #69: The { trick causes AI to omit the final }
                            # If we have opening brace but no closing, just add it!
                            print(f"  ⚠️  Could not find matching closing brace")
                            print(
                                f"  🔧 FALLBACK #69: Attempting to complete JSON by adding missing '}}' "
                            )

                            # Count how many braces are unmatched
                            open_count = json_str.count("{")
                            close_count = json_str.count("}")
                            missing_braces = open_count - close_count

                            if missing_braces > 0:
                                print(
                                    f"  Missing {missing_braces} closing brace(s), adding them..."
                                )
                                json_str = json_str + ("}" * missing_braces)
                                print(
                                    f"  ✓ Completed JSON! New last 40 chars: ...{repr(json_str[-40:])}"
                                )
                            else:
                                print(f"  ❌ Brace count mismatch - cannot auto-fix")

                    # CRITICAL: Unescape JSON BEFORE checking if it's valid
                    # The API sometimes returns escaped JSON like \{ instead of {
                    if (
                        "\\{" in json_str
                        or "\\}" in json_str
                        or "\\[" in json_str
                        or "\\]" in json_str
                    ):
                        print("Detected escaped JSON, unescaping...")
                        json_str = json_str.replace("\\{", "{").replace("\\}", "}")
                        json_str = json_str.replace("\\[", "[").replace("\\]", "]")
                        json_str = json_str.replace('\\"', '"')  # Unescape quotes too

                    if json_str.startswith("{") and json_str.endswith("}"):
                        print(
                            f"✓ JSON validation passed - attempting parse (length: {len(json_str)} chars)"
                        )
                        print(f"  First 100 chars: {json_str[:100]}")
                        print(f"  Last 100 chars: {json_str[-100:]}")
                        parsed_result = json.loads(json_str)

                        result = {
                            "success": True,
                            "data": parsed_result,
                            "raw_response": generated_text,
                        }
                        # Add warning if we had to aggressively trim the response
                        if used_aggressive_trimming:
                            result["parsing_warning"] = (
                                f"AI returned malformed response. Had to cut off {chars_cut} characters to extract JSON."
                            )
                        return result
                    else:
                        # FALLBACK: Try to extract data from non-JSON formats
                        print(f"✗ JSON validation failed!")
                        print(f"  Starts with '{{': {json_str.startswith('{')}")
                        print(f"  Ends with '}}': {json_str.endswith('}')}")
                        print(f"  First 50 chars: {json_str[:50]}")
                        print(f"  Last 50 chars: {json_str[-50:]}")
                        print(f"Attempting fallback extraction...")
                        fallback_data = self._extract_fallback_data(
                            generated_text, multiple_objects_data
                        )
                        if fallback_data:
                            print(f"Fallback extraction successful!")
                            return {
                                "success": True,
                                "data": fallback_data,
                                "raw_response": generated_text,
                                "parsing_warning": "AI returned non-JSON format. Fallback extraction was used.",
                            }
                        else:
                            return {
                                "success": False,
                                "error": "Could not find valid JSON in response",
                                "raw_response": generated_text,
                            }

                except json.JSONDecodeError as e:
                    # FALLBACK: Try to extract data from bullet points or other formats
                    print(f"✗ JSON parsing failed: {str(e)}")
                    print(
                        f"  Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}"
                    )
                    print(f"  Attempting fallback extraction...")
                    fallback_data = self._extract_fallback_data(
                        generated_text, multiple_objects_data
                    )
                    if fallback_data:
                        print(f"Fallback extraction successful!")
                        return {
                            "success": True,
                            "data": fallback_data,
                            "raw_response": generated_text,
                            "parsing_warning": "AI returned non-JSON format. Fallback extraction was used.",
                        }
                    else:
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

            # If 401, refresh token and retry once
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

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

            # If 401, refresh token and retry once
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=30
                )

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

    def _extract_fallback_data(
        self, text: str, multiple_objects_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Fallback parser to extract data from non-JSON AI responses (e.g., bullet points).
        Returns a valid data structure or None if extraction fails.
        """
        import re

        try:
            is_multi = multiple_objects_data is not None and multiple_objects_data.get(
                "multiple_objects", False
            )
            object_count = multiple_objects_data.get("count", 1) if is_multi else 1

            if is_multi:
                # Multi-object fallback parsing
                print(
                    f"Attempting multi-object fallback extraction for {object_count} objects..."
                )

                # Initialize arrays for each field
                fields = {
                    "condition": [],
                    "colour": [],
                    "material": [],
                    "category": [],
                    "brand": [],
                    "model": [],
                    "quantity": [],
                    "additional_info": [],
                }
                confidence = {
                    "condition": [],
                    "colour": [],
                    "material": [],
                    "category": [],
                    "brand": [],
                    "model": [],
                    "quantity": [],
                }
                descriptions = []

                # Try to extract object-by-object data
                object_patterns = [
                    r"\*\*Object (\d+):[^*]+\*\*",  # **Object 1: Name**
                    r"Object (\d+):",  # Object 1:
                ]

                # Find all object sections
                for obj_idx in range(object_count):
                    obj_num = obj_idx + 1

                    # Extract fields for this object using various patterns
                    condition = self._extract_field(
                        text,
                        f"Object {obj_num}",
                        ["Condition", "condition"],
                        ["used", "new", "open_box", "damaged", "unknown"],
                    )
                    colour = self._extract_field(
                        text,
                        f"Object {obj_num}",
                        ["Colour", "Color", "colour", "color"],
                        [
                            "red",
                            "blue",
                            "black",
                            "white",
                            "gray",
                            "silver",
                            "gold",
                            "green",
                            "brown",
                            "beige",
                            "transparent",
                            "unknown",
                        ],
                    )
                    material = self._extract_field(
                        text,
                        f"Object {obj_num}",
                        ["Material", "material"],
                        [
                            "plastic",
                            "metal",
                            "textile",
                            "paper",
                            "cardboard",
                            "wood",
                            "glass",
                            "ceramic",
                            "other",
                            "unknown",
                        ],
                    )
                    category = self._extract_field(
                        text,
                        f"Object {obj_num}",
                        ["Category", "category"],
                        [
                            "electronics",
                            "apparel",
                            "housewares",
                            "toys",
                            "tools",
                            "books",
                            "beauty",
                            "sports",
                            "other",
                            "unknown",
                        ],
                    )
                    brand = self._extract_field(
                        text, f"Object {obj_num}", ["Brand", "brand"], None
                    )
                    model = self._extract_field(
                        text, f"Object {obj_num}", ["Model", "model"], None
                    )
                    quantity = self._extract_field(
                        text, f"Object {obj_num}", ["Quantity", "quantity"], None
                    )

                    fields["condition"].append(condition or "unknown")
                    fields["colour"].append(colour or "unknown")
                    fields["material"].append(material or "unknown")
                    fields["category"].append(category or "unknown")
                    fields["brand"].append(brand or "")
                    fields["model"].append(model or "")
                    fields["quantity"].append(
                        int(quantity) if quantity and quantity.isdigit() else 1
                    )
                    fields["additional_info"].append("")

                    # Set default confidence
                    for key in confidence:
                        confidence[key].append(0.5)

                    descriptions.append(
                        f"Object {obj_num}: {brand or 'Unknown'} {model or 'item'}"
                    )

                return {
                    "fields": fields,
                    "confidence": confidence,
                    "descriptions": descriptions,
                    "global": {
                        "object_count": object_count,
                        "print_label": False,
                        "ocr_text": "",
                        "visible_marks": "",
                        "needs_review": True,  # Mark for review since this is fallback data
                    },
                }
            else:
                # Single-object fallback parsing
                print(f"Attempting single-object fallback extraction...")

                condition = self._extract_field(
                    text,
                    None,
                    ["Condition", "condition"],
                    ["used", "new", "open_box", "damaged", "unknown"],
                )
                colour = self._extract_field(
                    text,
                    None,
                    ["Colour", "Color", "colour", "color"],
                    [
                        "red",
                        "blue",
                        "black",
                        "white",
                        "gray",
                        "silver",
                        "gold",
                        "green",
                        "brown",
                        "beige",
                        "transparent",
                        "unknown",
                    ],
                )
                material = self._extract_field(
                    text,
                    None,
                    ["Material", "material"],
                    [
                        "plastic",
                        "metal",
                        "textile",
                        "paper",
                        "cardboard",
                        "wood",
                        "glass",
                        "ceramic",
                        "other",
                        "unknown",
                    ],
                )
                category = self._extract_field(
                    text,
                    None,
                    ["Category", "category"],
                    [
                        "electronics",
                        "apparel",
                        "housewares",
                        "toys",
                        "tools",
                        "books",
                        "beauty",
                        "sports",
                        "other",
                        "unknown",
                    ],
                )
                brand = self._extract_field(text, None, ["Brand", "brand"], None)
                model = self._extract_field(text, None, ["Model", "model"], None)
                quantity = self._extract_field(
                    text, None, ["Quantity", "quantity"], None
                )

                return {
                    "fields": {
                        "condition": condition or "unknown",
                        "colour": colour or "unknown",
                        "colour_secondary": {"color": "null", "percentage": 0},
                        "material": material or "unknown",
                        "material_secondary": {"material": "null", "percentage": 0},
                        "category": category or "unknown",
                        "brand": brand or "",
                        "model": model or "",
                        "quantity": (
                            int(quantity) if quantity and quantity.isdigit() else 1
                        ),
                        "weight": {"value": None, "unit": "null"},
                        "print_label": False,
                        "sort": "vague_overgoods",
                        "additional_info": "",
                    },
                    "confidence": {
                        "condition": 0.5,
                        "colour": 0.5,
                        "colour_secondary": 0.0,
                        "material": 0.5,
                        "material_secondary": 0.0,
                        "category": 0.5,
                        "brand": 0.5,
                        "model": 0.5,
                        "quantity": 0.5,
                        "weight": 0.0,
                        "print_label": 0.5,
                        "sort": 0.5,
                    },
                    "description": f"{brand or 'Unknown'} {model or 'item'}",
                    "evidence": {"ocr_like_text": "", "visible_marks": ""},
                    "needs_review": True,  # Mark for review since this is fallback data
                }

        except Exception as e:
            print(f"Fallback extraction error: {e}")
            return None

    def _extract_field(
        self,
        text: str,
        object_prefix: str,
        field_names: list,
        valid_values: list = None,
    ) -> str:
        """
        Helper to extract a field value from text using various patterns.
        """
        import re

        for field_name in field_names:
            # Pattern 1: **Field:** value or * **Field:** value
            if object_prefix:
                pattern = (
                    rf"{object_prefix}[^\n]*?[\*\s]*{field_name}[\*\s]*:[\s]*([^\n]+)"
                )
            else:
                pattern = rf"[\*\s]*{field_name}[\*\s]*:[\s]*([^\n]+)"

            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip("*").strip()

                # Clean up LaTeX artifacts that sometimes appear in AI responses
                # Remove patterns like "} \\", "\text{", "\\", etc.
                value = re.sub(r"\}\s*\\+", "", value)  # Remove "} \\" or "} \\\\"
                value = re.sub(
                    r"\\text\{([^}]*)\}", r"\1", value
                )  # Extract from \text{...}
                value = re.sub(r"\\+$", "", value)  # Remove trailing backslashes
                value = re.sub(r"\s+", " ", value).strip()  # Normalize whitespace

                # If valid_values provided, check if extracted value is in the list
                if valid_values:
                    value_lower = value.lower()
                    for valid_val in valid_values:
                        if valid_val.lower() in value_lower:
                            return valid_val
                else:
                    return value

        return None

    def generate_refinement_questions(
        self, discriminators: List[Dict[str, Any]], item_category: str = "item"
    ) -> Dict[str, Any]:
        """Generate natural language questions to help narrow down search results"""
        try:
            # Get access token
            access_token = self.get_access_token()

            # Build a description of the discriminators
            discriminator_descriptions = []
            for i, disc in enumerate(discriminators[:5], 1):  # Max 5 questions
                field = disc["field"]
                values = disc["values"]
                discriminator_descriptions.append(
                    f"{i}. **{field.capitalize()}**: Options are {', '.join(values)}"
                )

            discriminators_text = "\n".join(discriminator_descriptions)

            # Prompt to generate questions
            prompt = f"""CRITICAL FORMAT RULE: Your response must be PURE JSON starting with {{ and ending with }}. No markdown code blocks, no "Answer:", no "JSON Output:", no text before or after the JSON. Output ONLY the JSON object.

You are helping a warehouse clerk identify the correct item from search results. The clerk searched for "{item_category}" and got multiple results. You need to generate 3-5 natural, conversational questions that will help identify the exact item they're looking for.

Available discriminating features:
{discriminators_text}

IMPORTANT GUIDELINES:
1. Questions should be natural and conversational (e.g., "What color is it?" not "Select color:")
2. Keep questions SHORT - avoid listing all options in the question text
3. Order questions from most to least discriminating (brand/category first, then specifics)
4. For model questions, just ask "What model is it?" - DON'T list all models in the question
5. Questions are OPTIONAL - the user may skip any they don't know

Schema:

{{
  "questions": [
    {{
      "question": "Short, natural question (e.g., 'What brand is it?' or 'What color is it?')",
      "field": "field_name",
      "options": ["option1", "option2", "option3"]
    }},
    ...
  ]
}}

Generate {min(len(discriminators), 5)} questions maximum. Keep question text under 50 characters.

Begin your response with {{ now:
{{"""

            # Prepare API request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            payload = {
                "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 1000,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                "project_id": self.project_id,
            }

            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            # Handle token expiration
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                generated_text = (
                    result["choices"][0]["message"]["content"]
                    if "choices" in result
                    else result.get("results", [{}])[0].get("generated_text", "")
                )

                # Parse JSON response
                try:
                    json_str = generated_text.strip()

                    # Strip markdown code blocks
                    if "```" in json_str:
                        code_block_match = re.search(
                            r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL
                        )
                        if code_block_match:
                            json_str = code_block_match.group(1).strip()

                    # Extract JSON
                    if not json_str.startswith("{"):
                        start_idx = json_str.find("{")
                        if start_idx != -1:
                            json_str = json_str[start_idx:]

                    if not json_str.endswith("}"):
                        end_idx = json_str.rfind("}") + 1
                        if end_idx > 0:
                            json_str = json_str[:end_idx]

                    # Unescape
                    if "\\{" in json_str or "\\}" in json_str:
                        json_str = json_str.replace("\\{", "{").replace("\\}", "}")
                        json_str = json_str.replace("\\[", "[").replace("\\]", "]")
                        json_str = json_str.replace('\\"', '"')

                    parsed_result = json.loads(json_str)

                    return {
                        "success": True,
                        "questions": parsed_result.get("questions", []),
                    }

                except json.JSONDecodeError as e:
                    print(f"Failed to parse questions JSON: {e}")
                    return {
                        "success": False,
                        "error": f"JSON parsing error: {str(e)}",
                    }
            else:
                return {
                    "success": False,
                    "error": f"API call failed: {response.status_code}",
                }

        except Exception as e:
            print(f"Error generating refinement questions: {e}")
            return {"success": False, "error": str(e)}

    def extract_invoice_data(self, invoice_path: str) -> Dict[str, Any]:
        """Extract structured data from invoice image"""
        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode image with correct MIME type
            image_base64 = self.encode_image(invoice_path)
            mime_type = self.get_image_mime_type(invoice_path)
            print(f"Image MIME type: {mime_type}")

            # Prompt to extract invoice data
            prompt = """CRITICAL FORMAT RULE: Your response must be PURE JSON starting with { and ending with }. No markdown code blocks, no "Answer:", no "JSON Output:", no text before or after the JSON. Output ONLY the JSON object.

You are analyzing an invoice image. Extract ALL relevant product information from this invoice.

IMPORTANT: Extract EXACTLY what you see on the invoice. Don't infer or guess.

Schema:

{
  "item_description": "Full item description from invoice",
  "brand": "Brand name (or null if not visible)",
  "model": "Model name/number (or null if not visible)",
  "color": "Color (or null if not visible)",
  "material": "Material (or null if not visible)",
  "condition": "Condition (new/used/refurbished, or null if not stated)",
  "category": "Item category (electronics, furniture, etc.)",
  "quantity": "Quantity",
  "confidence": "high/medium/low - how clear is the invoice text?"
}

RULES:
- Extract ONLY what's explicitly stated on the invoice
- If a field is not visible/mentioned, use null
- For condition: only set if explicitly stated (e.g., "New", "Used", "Refurbished")
- Be precise with brand and model names

Begin your response with { now:
{"""

            # Prepare API request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

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
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 1000,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                "project_id": self.project_id,
            }

            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            # Handle token expiration
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                generated_text = (
                    result["choices"][0]["message"]["content"]
                    if "choices" in result
                    else result.get("results", [{}])[0].get("generated_text", "")
                )

                print(f"\nInvoice API Response Status: {response.status_code}")
                print(f"Generated invoice data: {generated_text[:200]}...")

                # Parse JSON response
                try:
                    json_str = generated_text.strip()

                    # Strip markdown code blocks
                    if "```" in json_str:
                        code_block_match = re.search(
                            r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL
                        )
                        if code_block_match:
                            json_str = code_block_match.group(1).strip()

                    # Extract JSON if wrapped in text
                    if not json_str.startswith("{"):
                        start_idx = json_str.find("{")
                        if start_idx != -1:
                            json_str = json_str[start_idx:]

                    if not json_str.endswith("}"):
                        end_idx = json_str.rfind("}") + 1
                        if end_idx > 0:
                            json_str = json_str[:end_idx]

                    # Unescape JSON
                    if "\\{" in json_str or "\\}" in json_str:
                        json_str = (
                            json_str.replace("\\{", "{")
                            .replace("\\}", "}")
                            .replace("\\[", "[")
                            .replace("\\]", "]")
                            .replace('\\"', '"')
                        )

                    if json_str.startswith("{") and json_str.endswith("}"):
                        parsed_result = json.loads(json_str)
                        print("✓ Invoice data extracted successfully")
                        return {"success": True, "data": parsed_result}
                    else:
                        raise ValueError("Response is not valid JSON")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"JSON parsing failed for invoice: {e}")
                    return {
                        "success": False,
                        "error": "Failed to parse invoice data",
                        "raw_response": generated_text,
                    }
            else:
                error_body = response.text
                print(f"❌ API call failed with status {response.status_code}")
                print(f"Response body: {error_body[:500]}")

                # More user-friendly error message
                if response.status_code == 500:
                    error_msg = "Watsonx AI service is temporarily unavailable (IBM server error). Please try again in a moment."
                elif response.status_code == 429:
                    error_msg = (
                        "Rate limit exceeded. Please wait a moment and try again."
                    )
                else:
                    error_msg = f"AI service returned error {response.status_code}"

                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": error_body,
                }

        except Exception as e:
            print(f"❌ Error extracting invoice data: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def verify_invoice_against_item(
        self, invoice_path: str, item_description: str, item_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask AI to verify if invoice matches the product description (holistic judgment)"""
        try:
            # Get access token
            access_token = self.get_access_token()

            # Encode invoice image
            image_base64 = self.encode_image(invoice_path)
            mime_type = self.get_image_mime_type(invoice_path)

            # Build item info for comparison
            item_info = f"""
Item Description: {item_description}
Brand: {item_metadata.get('brand', 'Not identified')}
Model: {item_metadata.get('model', 'Not identified')}
Color: {item_metadata.get('color', 'Unknown')}
Material: {item_metadata.get('material', 'Unknown')}
Condition: {item_metadata.get('condition', 'Unknown')}
Category: {item_metadata.get('category', 'Unknown')}
"""

            # Prompt for holistic verification
            prompt = f"""CRITICAL FORMAT RULE: Your response must be PURE JSON starting with {{ and ending with }}. No markdown, no text before/after.

You are analyzing documents for a warehouse inventory system. Compare this invoice/document image against the following product information:

{item_info}

**YOUR TASK:**
Determine if this document LIKELY corresponds to the product described above. Consider:
- Variations in naming (e.g., "Xbox Controller" vs "Microsoft Wireless Controller")
- Similar but not exact descriptions
- Context clues (packaging, branding, visual appearance)
- Be LENIENT - products can be described differently on documents vs retail descriptions

**IMPORTANT:**
- If the document clearly shows a DIFFERENT product type, that's a mismatch
- If details are similar or compatible, that's likely a match
- If you're uncertain due to missing information, err on the side of "possible match"

Return this exact JSON format (example with actual boolean):

{{
  "matches": true,
  "confidence": "high",
  "reasoning": "Brief explanation of why they match or don't match",
  "invoice_product": "What product is shown on the document",
  "key_discrepancies": []
}}

FIELD REQUIREMENTS: 
- "matches" must be boolean true or false (no quotes)
- "confidence" must be string "high", "medium", or "low"
- "key_discrepancies" must be array, use [] if none

Begin your response with {{ now:
{{"""

            # API request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

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
                                    "url": f"data:{mime_type};base64,{image_base64}"
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

            api_url = f"{self.url}/ml/v1/text/chat?version=2023-05-29"

            print(f"\n🔍 Asking AI to verify invoice against item...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            # Handle token expiration
            if response.status_code == 401:
                print("Token expired, refreshing...")
                access_token = self.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {access_token}"
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )

            if response.status_code == 200:
                result = response.json()
                generated_text = (
                    result["choices"][0]["message"]["content"]
                    if "choices" in result
                    else result.get("results", [{}])[0].get("generated_text", "")
                )

                print(f"AI verification response: {generated_text[:200]}...")

                # Parse JSON response (using our ROBUST parsing with Fallback #69!)
                try:
                    json_str = generated_text.strip()
                    print(
                        f"After .strip(): length={len(json_str)}, last char='{json_str[-1]}' (ASCII {ord(json_str[-1])})"
                    )

                    # Track if we used fallback parsing
                    used_fallback = False
                    chars_cut = 0

                    # Handle markdown code blocks
                    if "```" in json_str:
                        code_block_match = re.search(
                            r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL
                        )
                        if code_block_match:
                            json_str = code_block_match.group(1).strip()
                            print(f"Extracted JSON from markdown code block")
                            used_fallback = True

                    # If it starts with text before JSON, extract just the JSON part
                    if not json_str.startswith("{"):
                        start_idx = json_str.find("{")
                        if start_idx != -1:
                            print(
                                f"  Trimming start - found '{{' at position {start_idx}"
                            )
                            json_str = json_str[start_idx:]

                    # If it has text after JSON, extract just the JSON part
                    # We need to find the MATCHING closing brace, not just use rfind()
                    original_len = len(json_str)
                    if not json_str.endswith("}"):
                        last_char = json_str[-1]
                        print(
                            f"  ⚠️  JSON doesn't end with '}}' - last char: '{last_char}' (ASCII: {ord(last_char)})"
                        )
                        print(f"  Last 40 chars: ...{repr(json_str[-40:])}")

                        # Find the matching closing brace by counting nesting levels
                        brace_count = 0
                        end_idx = -1
                        for i, char in enumerate(json_str):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found the matching closing brace for the first opening brace
                                    end_idx = i + 1
                                    break

                        if end_idx > 0:
                            print(
                                f"  ✓ Found MATCHING '}}' at position: {end_idx - 1} (will cut to length {end_idx})"
                            )
                            json_str = json_str[:end_idx]
                            chars_cut = original_len - len(json_str)
                            print(
                                f"  ✂️  CUT OFF {chars_cut} chars! ({original_len} → {len(json_str)})"
                            )
                            print(
                                f"  After trim, last 40 chars: ...{repr(json_str[-40:])}"
                            )
                            # If we cut more than 100 chars, this is aggressive trimming
                            if chars_cut > 100:
                                used_fallback = True
                        else:
                            # FALLBACK #69: The { trick causes AI to omit the final }
                            print(f"  ⚠️  Could not find matching closing brace")
                            print(
                                f"  🔧 FALLBACK #69: Attempting to complete JSON by adding missing '}}' "
                            )

                            # Count how many braces are unmatched
                            open_count = json_str.count("{")
                            close_count = json_str.count("}")
                            missing_braces = open_count - close_count

                            if missing_braces > 0:
                                print(
                                    f"  Missing {missing_braces} closing brace(s), adding them..."
                                )
                                json_str = json_str + ("}" * missing_braces)
                                print(
                                    f"  ✓ Completed JSON! New last 40 chars: ...{repr(json_str[-40:])}"
                                )
                            else:
                                print(f"  ❌ Brace count mismatch - cannot auto-fix")

                    # CRITICAL: Unescape JSON BEFORE checking if it's valid
                    if (
                        "\\{" in json_str
                        or "\\}" in json_str
                        or "\\[" in json_str
                        or "\\]" in json_str
                    ):
                        print("Detected escaped JSON, unescaping...")
                        json_str = json_str.replace("\\{", "{").replace("\\}", "}")
                        json_str = json_str.replace("\\[", "[").replace("\\]", "]")
                        json_str = json_str.replace('\\"', '"')

                    if json_str.startswith("{") and json_str.endswith("}"):
                        print(
                            f"✓ JSON validation passed - attempting parse (length: {len(json_str)} chars)"
                        )
                        parsed_result = json.loads(json_str)
                        print(
                            f"✓ AI says: matches={parsed_result.get('matches')}, confidence={parsed_result.get('confidence')}"
                        )
                        result = {"success": True, "verification": parsed_result}
                        if used_fallback:
                            if chars_cut > 0:
                                result["parsing_warning"] = (
                                    f"AI returned malformed response. Had to cut off {chars_cut} characters to extract JSON."
                                )
                            else:
                                result["parsing_warning"] = (
                                    "AI returned non-JSON format. Fallback parsing was used."
                                )
                        return result
                    else:
                        print(f"✗ JSON validation failed!")
                        print(f"  Starts with '{{': {json_str.startswith('{')}")
                        print(f"  Ends with '}}': {json_str.endswith('}')}")
                        print(f"  First 50 chars: {json_str[:50]}")
                        print(f"  Last 50 chars: {json_str[-50:]}")
                        raise ValueError("Response is not valid JSON")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"✗ JSON parsing failed: {str(e)}")
                    print(
                        f"  Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}"
                    )
                    return {
                        "success": False,
                        "error": "Failed to parse verification response",
                        "raw_response": generated_text,
                    }
            else:
                error_msg = f"AI service returned error {response.status_code}"
                return {"success": False, "error": error_msg}

        except Exception as e:
            print(f"❌ Error verifying invoice: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}
