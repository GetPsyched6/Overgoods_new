import json
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from app.core import config


class SimpleVectorDatabase:
    """Simplified vector database using file-based storage and text similarity"""

    def __init__(self):
        """Initialize simple file-based storage"""
        self.db_file = os.path.join(config.CHROMA_DB_PATH, "items.json")
        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
        self.items = self._load_items()

    def _load_items(self) -> List[Dict[str, Any]]:
        """Load items from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_items(self):
        """Save items to JSON file"""
        with open(self.db_file, "w") as f:
            json.dump(self.items, f, indent=2)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using improved word-based matching optimized for search"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Basic sequence matching as baseline
        sequence_sim = SequenceMatcher(None, text1_lower, text2_lower).ratio()

        # Word-based similarity for better semantic matching
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        if not words1 or not words2:
            return sequence_sim

        # Calculate intersection-based similarity (favor recall over precision for search)
        intersection = len(words1.intersection(words2))

        # Use the smaller set as denominator to favor queries that are subsets of descriptions
        min_words = min(len(words1), len(words2))
        intersection_sim = intersection / min_words if min_words > 0 else 0

        # Also calculate traditional Jaccard for balance
        union = len(words1.union(words2))
        jaccard_sim = intersection / union if union > 0 else 0

        # Weighted combination: favor intersection-based for search queries
        combined_sim = 0.2 * sequence_sim + 0.6 * intersection_sim + 0.2 * jaccard_sim

        # Strong boost for exact brand/model matches
        important_terms = [
            "pioneer",
            "cdj",
            "xbox",
            "controller",
            "siemens",
            "corel",
            "allen-bradley",
            "anthem",
            "videostudio",
        ]
        boost_count = 0
        for term in important_terms:
            if term in text1_lower and term in text2_lower:
                boost_count += 1

        # Apply boost based on number of important terms matched
        if boost_count > 0:
            combined_sim += 0.3 * boost_count  # Stronger boost for multiple matches

        return min(combined_sim, 1.0)  # Cap at 1.0

    def add_item(
        self, item_id: str, description: str, image_path: str, metadata: Dict[str, Any]
    ) -> bool:
        """Add an item to the database"""
        try:
            item = {
                "id": item_id,
                "description": description,
                "image_path": image_path,
                "metadata": metadata,
            }

            # Remove existing item with same ID
            self.items = [item for item in self.items if item["id"] != item_id]

            # Add new item
            self.items.append(item)
            self._save_items()
            return True
        except Exception as e:
            print(f"Error adding item to database: {e}")
            return False

    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search items by text description using similarity matching"""
        try:
            results = []

            for item in self.items:
                # Calculate similarity with description
                desc_similarity = self._calculate_similarity(query, item["description"])

                # Calculate similarity with metadata fields
                metadata_text = " ".join(
                    [
                        str(item["metadata"].get("title", "")),
                        str(item["metadata"].get("color", "")),
                        str(item["metadata"].get("material", "")),
                        str(item["metadata"].get("condition", "")),
                        str(item["metadata"].get("category", "")),
                    ]
                )
                meta_similarity = self._calculate_similarity(query, metadata_text)

                # Use the higher similarity score
                similarity = max(desc_similarity, meta_similarity)

                results.append(
                    {
                        "id": item["id"],
                        "description": item["description"],
                        "metadata": item["metadata"],
                        "distance": 1 - similarity,  # Convert similarity to distance
                        "similarity": similarity,
                    }
                )

            # Sort by similarity (descending) and return top n_results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:n_results]

        except Exception as e:
            print(f"Error searching by text: {e}")
            return []

    def search_by_form_data(
        self, form_data: Dict[str, str], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search items by form data"""
        # Build search query from form data
        query_parts = []

        if form_data.get("keywords"):
            query_parts.append(form_data["keywords"])
        if form_data.get("color"):
            query_parts.append(f"color {form_data['color']}")
        if form_data.get("material"):
            query_parts.append(f"material {form_data['material']}")
        if form_data.get("condition"):
            query_parts.append(f"condition {form_data['condition']}")
        if form_data.get("category"):
            query_parts.append(f"category {form_data['category']}")

        query = " ".join(query_parts) if query_parts else "general item"

        return self.search_by_text(query, n_results)

    def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items in the database"""
        try:
            return [
                {
                    "id": item["id"],
                    "description": item["description"],
                    "metadata": item["metadata"],
                }
                for item in self.items
            ]
        except Exception as e:
            print(f"Error getting all items: {e}")
            return []

    def delete_item(self, item_id: str) -> bool:
        """Delete an item from the database"""
        try:
            original_count = len(self.items)
            self.items = [item for item in self.items if item["id"] != item_id]

            if len(self.items) < original_count:
                self._save_items()
                return True
            return False
        except Exception as e:
            print(f"Error deleting item: {e}")
            return False

    def initialize_sample_data(self):
        """Initialize database with cached data or basic fallback data"""
        # Check if we have cached AI descriptions
        if self.items:
            print(f"✅ Loaded {len(self.items)} items from cache")
            return

        # Check if assets folder exists
        assets_dir = config.ASSETS_DIR
        if not os.path.exists(assets_dir):
            print("No assets folder found, using basic sample data")
            self._initialize_basic_sample_data()
            return

        # Get image files
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            image_files.extend(
                [f for f in os.listdir(assets_dir) if f.lower().endswith(ext)]
            )

        if not image_files:
            print("No images found in assets folder, using basic sample data")
            self._initialize_basic_sample_data()
            return

        # Create basic entries for images (without AI descriptions)
        print(f"📁 Found {len(image_files)} images, creating basic entries")
        for i, image_file in enumerate(image_files[:10], 1):
            item_id = f"asset_{i:03d}"
            image_path = os.path.join(assets_dir, image_file)

            # Basic description without AI
            description = f"Overgood item from {image_file} - AI description pending"
            metadata = {
                "title": f"Item {i}",
                "color": "Various",
                "material": "Mixed",
                "condition": "Unknown",
                "category": "Other",
                "sub_category": "Mixed Items",
                "image_path": image_path,
                "ai_generated": False,
            }

            self.add_item(item_id, description, image_path, metadata)

        print(
            f"✅ Created {len(image_files)} basic entries. Use 'Reprocess with AI' to generate detailed descriptions."
        )

    def generate_ai_descriptions(self):
        """Generate AI descriptions for all items (can be called separately)"""
        from app.services.watsonx_service import SimpleWatsonxClient

        try:
            ai_client = SimpleWatsonxClient()
            print("🤖 Generating AI descriptions for your images...")

            updated_count = 0
            for item in self.items:
                if not item["metadata"].get("ai_generated", False):
                    image_path = item["image_path"]
                    print(f"Processing {os.path.basename(image_path)}...")

                    ai_result = ai_client.generate_search_embedding(image_path)

                    if ai_result["success"]:
                        # Update with AI-generated description
                        description = ai_result["description"]
                        desc_lower = description.lower()

                        # Extract category and condition from description
                        category, sub_category = self._extract_category(desc_lower)
                        condition = self._extract_condition(desc_lower)

                        # Update the item
                        item["description"] = description
                        item["metadata"].update(
                            {
                                "title": f"{item['id'].split('_')[1]} - {category}",
                                "condition": condition,
                                "category": category,
                                "sub_category": sub_category,
                                "ai_generated": True,
                            }
                        )
                        updated_count += 1
                    else:
                        print(
                            f"AI description failed for {os.path.basename(image_path)}"
                        )

            # Save updated items
            self._save_items()
            print(f"✅ Updated {updated_count} items with AI descriptions!")
            return {"success": True, "updated": updated_count}

        except Exception as e:
            print(f"❌ AI description generation failed: {e}")
            return {"success": False, "error": str(e)}

    def _extract_category(self, desc_lower):
        """Extract category from description"""
        if any(
            word in desc_lower
            for word in [
                "electronic",
                "device",
                "cd",
                "player",
                "audio",
                "speaker",
                "headphone",
                "cable",
            ]
        ):
            return "Electronics", "Audio/Video"
        elif any(
            word in desc_lower
            for word in ["clothing", "shirt", "pants", "jacket", "fabric", "textile"]
        ):
            return "Clothing", "Apparel"
        elif any(
            word in desc_lower
            for word in ["game", "controller", "xbox", "playstation", "gaming"]
        ):
            return "Electronics", "Gaming"
        elif any(word in desc_lower for word in ["book", "media", "dvd", "cd", "disc"]):
            return "Media", "Entertainment"
        elif any(
            word in desc_lower
            for word in ["care", "hygiene", "beauty", "cosmetic", "shampoo"]
        ):
            return "Personal Care", "Health & Beauty"
        else:
            return "Other", "Mixed Items"

    def _extract_condition(self, desc_lower):
        """Extract condition from description"""
        if any(word in desc_lower for word in ["new", "unopened", "sealed"]):
            return "New"
        elif any(word in desc_lower for word in ["used", "worn", "opened"]):
            return "Used"
        elif any(word in desc_lower for word in ["damaged", "broken", "cracked"]):
            return "Damaged"
        else:
            return "Unknown"

    def _initialize_basic_sample_data(self):
        """Initialize with basic sample data"""
        sample_items = [
            {
                "id": "sample_001",
                "description": "Black leather jacket, size medium, good condition",
                "image_path": "sample_images/jacket.jpg",
                "metadata": {
                    "title": "Leather Jacket",
                    "color": "Black",
                    "material": "Leather",
                    "condition": "Good",
                    "category": "Clothing",
                    "sub_category": "Outerwear",
                    "image_path": "sample_images/jacket.jpg",
                    "ai_generated": False,
                },
            },
            {
                "id": "sample_002",
                "description": "Wireless bluetooth headphones, noise cancelling",
                "image_path": "sample_images/headphones.jpg",
                "metadata": {
                    "title": "Bluetooth Headphones",
                    "color": "Black",
                    "material": "Plastic",
                    "condition": "New",
                    "category": "Electronics",
                    "sub_category": "Audio",
                    "image_path": "sample_images/headphones.jpg",
                    "ai_generated": False,
                },
            },
        ]

        for item in sample_items:
            self.add_item(
                item["id"],
                item["description"],
                item["image_path"],
                item["metadata"],
            )

    def _initialize_fallback_data(self, image_files, assets_dir):
        """Fallback method for when AI description generation fails"""
        for i, image_file in enumerate(image_files[:10], 1):
            item_id = f"asset_{i:03d}"
            image_path = os.path.join(assets_dir, image_file)

            description = f"Overgood item {i} from {image_file}"
            metadata = {
                "title": f"Item {i}",
                "color": "Various",
                "material": "Mixed",
                "condition": "Unknown",
                "category": "Other",
                "sub_category": "Mixed Items",
                "image_path": image_path,
                "ai_generated": False,
            }

            self.add_item(item_id, description, image_path, metadata)
