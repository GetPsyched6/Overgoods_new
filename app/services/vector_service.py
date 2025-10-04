import json
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import numpy as np
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
        self,
        item_id: str,
        description: str,
        image_path: str,
        metadata: Dict[str, Any],
        image_embedding: Optional[List[float]] = None,
    ) -> bool:
        """Add an item to the database

        Args:
            item_id: Unique identifier
            description: Text description
            image_path: Path to image file
            metadata: Item metadata dict
            image_embedding: Optional CLIP embedding vector (512 floats)
        """
        try:
            item = {
                "id": item_id,
                "description": description,
                "image_path": image_path,
                "metadata": metadata,
            }

            # Add embedding if provided
            if image_embedding is not None:
                item["image_embedding"] = image_embedding

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

                # Use weighted average favoring metadata for structured searches
                # If query contains field keywords like "color", "material", favor metadata more
                query_lower = query.lower()
                is_structured = any(
                    keyword in query_lower
                    for keyword in ["color", "material", "condition", "category"]
                )

                if is_structured:
                    # For structured queries, heavily favor metadata matches
                    similarity = 0.3 * desc_similarity + 0.7 * meta_similarity
                else:
                    # For free-text queries, balance both
                    similarity = 0.6 * desc_similarity + 0.4 * meta_similarity

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
        """Search items by form data with field-specific scoring"""
        try:
            results = []

            # Weights for each field (higher = more important)
            weights = {
                "color": 0.35,  # Color is very specific and distinctive
                "category": 0.25,  # Category is important for filtering
                "condition": 0.20,  # Condition matters for value
                "material": 0.15,  # Material is less distinctive
                "keywords": 0.05,  # Keywords as bonus
            }

            for item in self.items:
                score = 0.0
                matched_fields = 0
                total_fields = len([v for v in form_data.values() if v])

                # Score each field individually
                if form_data.get("color"):
                    item_color = item["metadata"].get("color", "").lower()
                    search_color = form_data["color"].lower()
                    if item_color == search_color:
                        score += weights["color"]
                        matched_fields += 1
                    elif item_color != "various" and search_color in item_color:
                        score += weights["color"] * 0.5  # Partial match

                if form_data.get("category"):
                    item_category = item["metadata"].get("category", "").lower()
                    search_category = form_data["category"].lower()
                    if item_category == search_category:
                        score += weights["category"]
                        matched_fields += 1
                    elif search_category in item_category:
                        score += weights["category"] * 0.7

                if form_data.get("condition"):
                    item_condition = item["metadata"].get("condition", "").lower()
                    search_condition = form_data["condition"].lower()
                    if item_condition == search_condition:
                        score += weights["condition"]
                        matched_fields += 1
                    elif (
                        item_condition != "unknown"
                        and search_condition in item_condition
                    ):
                        score += weights["condition"] * 0.5

                if form_data.get("material"):
                    item_material = item["metadata"].get("material", "").lower()
                    search_material = form_data["material"].lower()
                    if item_material == search_material:
                        score += weights["material"]
                        matched_fields += 1
                    elif item_material != "mixed" and search_material in item_material:
                        score += weights["material"] * 0.5
                    # Don't penalize if material is "mixed" (unknown)
                    elif item_material == "mixed":
                        matched_fields += 0.3  # Partial credit

                if form_data.get("keywords"):
                    keywords = form_data["keywords"].lower()
                    desc_lower = item["description"].lower()
                    if keywords in desc_lower:
                        score += weights["keywords"]

                # Bonus for matching multiple fields
                if total_fields > 0:
                    match_ratio = matched_fields / total_fields
                    score = score * (
                        0.7 + 0.3 * match_ratio
                    )  # 70-100% based on match ratio

                results.append(
                    {
                        "id": item["id"],
                        "description": item["description"],
                        "metadata": item["metadata"],
                        "distance": 1 - score,
                        "similarity": score,
                    }
                )

            # Sort by score (descending) and return top n_results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:n_results]

        except Exception as e:
            print(f"Error in form search: {e}")
            # Fallback to text search
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

    def search_by_image_embedding(
        self, query_embedding: np.ndarray, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search items using CLIP image embedding (visual similarity)

        Args:
            query_embedding: CLIP embedding vector of uploaded image (512 floats)
            n_results: Number of results to return

        Returns:
            List of matching items with similarity scores
        """
        try:
            from app.services.clip_service import CLIPService

            results = []

            # Collect all items with embeddings
            items_with_embeddings = [
                item
                for item in self.items
                if "image_embedding" in item and item["image_embedding"]
            ]

            if not items_with_embeddings:
                print(
                    "⚠️  No items have CLIP embeddings. Run embedding generation first."
                )
                return []

            # Convert embeddings to numpy array for batch processing
            embeddings_array = np.array(
                [item["image_embedding"] for item in items_with_embeddings]
            )

            # Calculate similarities (vectorized, super fast!)
            similarities = CLIPService.cosine_similarity_batch(
                query_embedding, embeddings_array
            )

            # Create results with scores
            for item, similarity in zip(items_with_embeddings, similarities):
                results.append(
                    {
                        "id": item["id"],
                        "description": item["description"],
                        "metadata": item["metadata"],
                        "similarity": float(similarity),
                        "distance": float(1 - similarity),
                    }
                )

            # Sort by similarity (descending) and return top n_results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:n_results]

        except Exception as e:
            print(f"Error in CLIP image search: {e}")
            return []

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

    def sync_with_assets(self) -> Dict[str, Any]:
        """Scan assets folder and add any new images to database

        Returns:
            Dict with success status and count of new items added
        """
        try:
            assets_dir = config.ASSETS_DIR
            if not os.path.exists(assets_dir):
                return {
                    "success": False,
                    "error": "Assets directory not found",
                    "added": 0,
                }

            # Get all image files in assets
            image_files = []
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                image_files.extend(
                    [f for f in os.listdir(assets_dir) if f.lower().endswith(ext)]
                )

            # Get existing image paths
            existing_paths = {item["image_path"] for item in self.items}

            # Find new images
            new_images = []
            for img_file in image_files:
                full_path = os.path.join(assets_dir, img_file)
                if full_path not in existing_paths:
                    new_images.append((img_file, full_path))

            if not new_images:
                return {"success": True, "added": 0, "total": len(self.items)}

            # Add new images
            added_count = 0
            for img_file, full_path in new_images:
                # Generate new ID
                existing_ids = {item["id"] for item in self.items}
                item_num = 1
                while f"asset_{item_num:03d}" in existing_ids:
                    item_num += 1

                item_id = f"asset_{item_num:03d}"

                # Create basic entry
                description = f"Item from {img_file} - AI description pending"
                metadata = {
                    "title": f"Item {item_num}",
                    "color": "Various",
                    "material": "Mixed",
                    "condition": "Unknown",
                    "category": "Other",
                    "sub_category": "Mixed Items",
                    "image_path": full_path,
                    "ai_generated": False,
                }

                self.add_item(item_id, description, full_path, metadata)
                added_count += 1
                print(f"✅ Added {item_id}: {img_file}")

            return {"success": True, "added": added_count, "total": len(self.items)}

        except Exception as e:
            print(f"❌ Error syncing assets: {e}")
            return {"success": False, "error": str(e), "added": 0}

    def generate_clip_embeddings(self, force_regenerate=False) -> Dict[str, Any]:
        """Generate CLIP embeddings for all items

        Args:
            force_regenerate: If True, regenerate embeddings for all items

        Returns:
            Dict with success status and counts
        """
        try:
            from app.services.clip_service import get_clip_service

            clip_service = get_clip_service()
            print("🎨 Generating CLIP embeddings...")

            total_items = len(self.items)
            pending_items = sum(
                1
                for item in self.items
                if not item.get("image_embedding") or force_regenerate
            )
            updated_count = 0

            if force_regenerate:
                print(f"Force regenerating embeddings for all {total_items} items")
            else:
                print(
                    f"Found {pending_items} items pending embeddings out of {total_items} total items"
                )

            for item in self.items:
                should_process = force_regenerate or not item.get("image_embedding")

                if should_process:
                    image_path = item["image_path"]
                    print(f"Processing {os.path.basename(image_path)}...")

                    try:
                        embedding = clip_service.encode_image(image_path)
                        item["image_embedding"] = embedding.tolist()
                        updated_count += 1
                        print(f"  ✓ Generated embedding")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")

            # Save updated items
            self._save_items()
            print(f"✅ Updated {updated_count} items with CLIP embeddings!")

            return {
                "success": True,
                "updated": updated_count,
                "total": total_items,
                "already_processed": (
                    total_items - pending_items if not force_regenerate else 0
                ),
                "force_regenerated": force_regenerate,
            }

        except Exception as e:
            print(f"❌ CLIP embedding generation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_ai_descriptions(self, force_regenerate=False):
        """Generate AI descriptions for all items (can be called separately)

        Args:
            force_regenerate: If True, regenerate descriptions for all items even if they already have AI descriptions
        """
        from app.services.watsonx_service import SimpleWatsonxClient

        try:
            ai_client = SimpleWatsonxClient()
            print("🤖 Generating AI descriptions for your images...")

            total_items = len(self.items)
            pending_items = sum(
                1
                for item in self.items
                if not item["metadata"].get("ai_generated", False)
            )
            updated_count = 0

            if force_regenerate:
                print(f"Force regenerating AI descriptions for all {total_items} items")
            else:
                print(
                    f"Found {pending_items} items pending AI descriptions out of {total_items} total items"
                )

            for item in self.items:
                should_process = force_regenerate or not item["metadata"].get(
                    "ai_generated", False
                )

                if should_process:
                    image_path = item["image_path"]
                    print(f"Processing {os.path.basename(image_path)}...")

                    # Step 1: Get structured metadata (color, material, condition, etc.)
                    print("  → Getting structured metadata...")
                    struct_result = ai_client.generate_product_description(image_path)

                    # Step 2: Get search-optimized text description
                    print("  → Getting search description...")
                    search_result = ai_client.generate_search_embedding(image_path)

                    if struct_result["success"] and search_result["success"]:
                        ai_data = struct_result["data"]

                        # Use the search-optimized description (better for matching)
                        description = search_result.get("description", "")
                        if not description:
                            # Fallback to structured API's description
                            description = ai_data.get(
                                "description", f"Item {item['id']}"
                            )

                        # Extract ACTUAL values from AI's structured fields
                        fields = ai_data.get("fields", {})

                        # Color
                        color = fields.get("colour", "Various")
                        if color in ["unknown", "null"]:
                            color = "Various"
                        else:
                            color = color.capitalize()

                        # Material
                        material = fields.get("material", "Mixed")
                        if material in ["unknown", "other", "null"]:
                            material = "Mixed"
                        else:
                            material = material.capitalize()

                        # Condition
                        condition = fields.get("condition", "Unknown")
                        condition_map = {
                            "new": "New",
                            "open_box": "Open Box",
                            "used": "Used",
                            "damaged": "Damaged",
                            "unknown": "Unknown",
                        }
                        condition = condition_map.get(condition.lower(), "Unknown")

                        # Category from AI or fallback to extraction
                        ai_category = fields.get("category", "").lower()
                        if ai_category in ["electronics", "electronic"]:
                            category, sub_category = "Electronics", "General"
                        elif ai_category in ["apparel", "clothing"]:
                            category, sub_category = "Clothing", "Apparel"
                        elif ai_category in ["housewares", "household"]:
                            category, sub_category = "Housewares", "General"
                        else:
                            # Fallback to extraction
                            desc_lower = description.lower()
                            category, sub_category = self._extract_category(desc_lower)

                        # Update the item with REAL AI data
                        item["description"] = description
                        item["metadata"].update(
                            {
                                "title": f"{item['id'].split('_')[1]} - {category}",
                                "color": color,
                                "material": material,
                                "condition": condition,
                                "category": category,
                                "sub_category": sub_category,
                                "brand": fields.get("brand", ""),
                                "model": fields.get("model", ""),
                                "ai_generated": True,
                            }
                        )
                        updated_count += 1
                        print(f"  ✓ Complete: {color} {material} - {condition}")
                    elif struct_result["success"]:
                        # At least got structured data, use it with basic description
                        print(
                            "  ⚠️  Search description failed, using structured data only"
                        )
                        ai_data = struct_result["data"]
                        fields = ai_data.get("fields", {})
                        description = ai_data.get("description", f"Item {item['id']}")

                        # Extract metadata as above (simplified for fallback)
                        color = fields.get("colour", "Various").capitalize()
                        material = fields.get("material", "Mixed").capitalize()
                        condition = fields.get("condition", "Unknown")
                        condition_map = {
                            "new": "New",
                            "open_box": "Open Box",
                            "used": "Used",
                            "damaged": "Damaged",
                            "unknown": "Unknown",
                        }
                        condition = condition_map.get(condition.lower(), "Unknown")

                        desc_lower = description.lower()
                        category, sub_category = self._extract_category(desc_lower)

                        item["description"] = description
                        item["metadata"].update(
                            {
                                "title": f"{item['id'].split('_')[1]} - {category}",
                                "color": color,
                                "material": material,
                                "condition": condition,
                                "category": category,
                                "sub_category": sub_category,
                                "brand": fields.get("brand", ""),
                                "model": fields.get("model", ""),
                                "ai_generated": True,
                            }
                        )
                        updated_count += 1
                        print(f"  ✓ Partial: {color} {material} - {condition}")
                    else:
                        print(
                            f"  ✗ Both APIs failed for {os.path.basename(image_path)}"
                        )

            # Save updated items
            self._save_items()
            print(f"✅ Updated {updated_count} items with AI descriptions!")
            return {
                "success": True,
                "updated": updated_count,
                "total": total_items,
                "already_processed": (
                    total_items - pending_items if not force_regenerate else 0
                ),
                "force_regenerated": force_regenerate,
            }

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

    def _extract_color(self, desc_lower):
        """Extract primary color from description"""
        # Color keywords to look for
        colors = {
            "purple": "Purple",
            "violet": "Purple",
            "black": "Black",
            "white": "White",
            "gray": "Gray",
            "grey": "Gray",
            "silver": "Silver",
            "red": "Red",
            "blue": "Blue",
            "green": "Green",
            "yellow": "Yellow",
            "orange": "Orange",
            "brown": "Brown",
            "beige": "Beige",
            "gold": "Gold",
            "pink": "Pink",
            "tan": "Tan",
        }

        # Find the first color mentioned
        for keyword, color_name in colors.items():
            if keyword in desc_lower:
                return color_name

        return "Various"

    def _extract_material(self, desc_lower):
        """Extract primary material from description (context-aware)"""
        # Material keywords to look for (ordered by priority for electronics)
        materials = {
            "plastic": "Plastic",
            "metal": "Metal",
            "aluminum": "Metal",
            "steel": "Metal",
            "iron": "Metal",
            "glass": "Glass",
            "ceramic": "Ceramic",
            "rubber": "Rubber",
            "silicon": "Rubber",
            "wood": "Wood",
            "wooden": "Wood",
            "fabric": "Fabric",
            "textile": "Fabric",
            "cloth": "Fabric",
            "paper": "Paper",
            "cardboard": "Cardboard",
            "leather": "Leather",
        }

        # Smart heuristics: ignore materials mentioned in background/packaging context
        # Match phrases that indicate the material is NOT the item itself
        ignore_contexts = [
            "background of the image",
            "background consists",
            "brown cardboard",
            "cardboard box",
            "cardboard packaging",
            "packaged in a cardboard",
            "wrapped in plastic",
            "plastic bag",
            "plastic insert",
            "leather desk",
            "leather surface",
            "leather table",
            "wooden desk",
            "wooden surface",
            "wooden table",
            "wood desk",
            "fabric background",
            "on a desk",
            "on a table",
            "on a surface",
        ]

        # For each material, check if it's mentioned in a valid context
        material_scores = {}
        for keyword, material_name in materials.items():
            if keyword not in desc_lower:
                continue

            # Find all occurrences
            count = desc_lower.count(keyword)

            # Check if it's in an ignore context
            in_valid_context = True
            for ignore_ctx in ignore_contexts:
                # Check if this material keyword appears near an ignore phrase
                if ignore_ctx in desc_lower:
                    # Find position of ignore context
                    ctx_pos = desc_lower.find(ignore_ctx)
                    # Find position of material keyword
                    mat_pos = desc_lower.find(keyword)
                    # If keyword is within 100 chars of ignore context, skip it
                    if abs(ctx_pos - mat_pos) < 100:
                        in_valid_context = False
                        break

            if in_valid_context:
                # Score based on position and count
                # Earlier mentions and multiple mentions score higher
                first_pos = desc_lower.find(keyword)
                score = count * 2 + (500 - first_pos) / 100  # Earlier = higher score
                material_scores[material_name] = score

        # Category-based defaults for electronics
        if (
            "electronic" in desc_lower
            or "device" in desc_lower
            or "controller" in desc_lower
        ):
            # Electronics are usually plastic or metal
            if "Plastic" in material_scores:
                material_scores["Plastic"] *= 1.5  # Boost plastic for electronics
            if "Metal" in material_scores:
                material_scores["Metal"] *= 1.3

        # Return the highest scoring material
        if material_scores:
            return max(material_scores.items(), key=lambda x: x[1])[0]

        # Smart defaults based on category
        if (
            "electronic" in desc_lower
            or "device" in desc_lower
            or "controller" in desc_lower
            or "console" in desc_lower
        ):
            return "Plastic"  # Most consumer electronics are plastic
        if "metal" in desc_lower or "industrial" in desc_lower:
            return "Metal"

        return "Mixed"

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
