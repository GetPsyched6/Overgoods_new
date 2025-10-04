from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os
import uuid
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.services.watsonx_service import SimpleWatsonxClient
from app.services.vector_service import SimpleVectorDatabase
from app.core.config import UPLOAD_DIR, ALLOWED_EXTENSIONS

router = APIRouter()

# Initialize clients
watsonx_client = SimpleWatsonxClient()
vector_db = SimpleVectorDatabase()


@router.on_event("startup")
async def startup_event():
    """Initialize the database with sample data"""
    vector_db.initialize_sample_data()


def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the file path"""
    # Validate file extension
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@router.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    with open("frontend/templates/home.html", "r") as f:
        return f.read()


@router.post("/api/reload-database")
async def reload_database():
    """Reload database from disk without restarting server"""
    try:
        result = vector_db.reload_from_disk()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/sync-assets")
async def sync_assets():
    """Scan assets folder and add any new images to database"""
    try:
        result = vector_db.sync_with_assets()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/generate-embeddings")
async def generate_embeddings(force: bool = Form(False)):
    """Generate CLIP embeddings for items

    Args:
        force: If True, regenerate embeddings for all items
    """
    try:
        result = vector_db.generate_clip_embeddings(force_regenerate=force)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/reprocess-descriptions")
async def reprocess_descriptions(force: bool = Form(False)):
    """Reprocess all items with AI descriptions

    Args:
        force: If True, regenerate descriptions for all items even if they already have AI descriptions
    """
    try:
        result = vector_db.generate_ai_descriptions(force_regenerate=force)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/check-multiple-objects")
async def check_multiple_objects(file: UploadFile = File(...)):
    """Check if image contains multiple objects"""
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file)

        # Check for multiple objects
        multiple_check = watsonx_client.check_multiple_objects(file_path)

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        if multiple_check["success"]:
            return JSONResponse(
                content={
                    "success": True,
                    "data": multiple_check["data"],
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": multiple_check["error"],
                },
            )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/generate-description")
async def generate_description(
    file: UploadFile = File(...), multiple_objects_json: str = Form(None)
):
    """Generate product description from uploaded image - just does the main analysis"""
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file)

        # Parse multiple objects data if provided
        multiple_objects_data = None
        if multiple_objects_json:
            import json

            multiple_objects_data = json.loads(multiple_objects_json)

        # Generate description using Watsonx (no multiple object check here - already done in Step 1)
        result = watsonx_client.generate_product_description(
            file_path, multiple_objects_data
        )

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        if result["success"]:
            return JSONResponse(
                content={
                    "success": True,
                    "data": result["data"],
                    "raw_response": result.get("raw_response", ""),
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result["error"],
                    "raw_response": result.get("raw_response", ""),
                },
            )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/search")
async def search_items(
    file: Optional[UploadFile] = File(None),
    keywords: Optional[str] = Form(None),
    natural_query: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    material: Optional[str] = Form(None),
    condition: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    n_results: int = Form(5),
):
    """Search for similar items using form data, natural language, and/or image"""
    try:
        results = []

        if file:
            # Image-based search using CLIP embeddings
            file_path = save_uploaded_file(file)

            try:
                # Try CLIP-based search (fast, accurate)
                from app.services.clip_service import get_clip_service

                clip_service = get_clip_service()
                query_embedding = clip_service.encode_image(file_path)
                results = vector_db.search_by_image_embedding(
                    query_embedding, n_results
                )

                print(f"🎯 CLIP image search completed: {len(results)} results")

                # If no results (no embeddings in DB), fall back to text-based search
                if not results:
                    print("⚠️  Falling back to text-based search...")
                    embedding_result = watsonx_client.generate_search_embedding(
                        file_path
                    )
                    if embedding_result["success"]:
                        results = vector_db.search_by_text(
                            embedding_result["description"], n_results
                        )
                        print(
                            f"Text search query: {embedding_result['description'][:100]}..."
                        )
                    else:
                        return JSONResponse(
                            status_code=500,
                            content={
                                "success": False,
                                "error": embedding_result["error"],
                            },
                        )

            except Exception as e:
                print(f"❌ CLIP search failed: {e}")
                # Fall back to text-based search
                embedding_result = watsonx_client.generate_search_embedding(file_path)
                if embedding_result["success"]:
                    results = vector_db.search_by_text(
                        embedding_result["description"], n_results
                    )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": str(e)},
                    )
            finally:
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
        elif natural_query:
            # Natural language search
            # For very short queries (1-2 words), skip NL processing for better precision
            query_words = natural_query.strip().split()
            is_short_query = len(query_words) <= 2 and len(natural_query) < 20

            if is_short_query:
                # Direct search for short queries (e.g., "cable", "laptop", "usb cable")
                # This avoids over-expansion and gives better results with frequency boost
                results = vector_db.search_by_text(natural_query, n_results)
                print(f"Direct search (short query): '{natural_query}'")
            else:
                # Use Watsonx NL processing for complex queries
                nl_result = watsonx_client.process_natural_language_search(
                    natural_query
                )

                if nl_result["success"]:
                    # Search using the processed natural language query
                    results = vector_db.search_by_text(
                        nl_result["processed_query"], n_results
                    )
                    print(
                        f"Natural language search: '{natural_query}' -> '{nl_result['processed_query'][:100]}...'"
                    )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": nl_result["error"]},
                    )
        else:
            # Form-based search
            form_data = {
                "keywords": keywords,
                "color": color,
                "material": material,
                "condition": condition,
                "category": category,
            }
            # Remove None values
            form_data = {k: v for k, v in form_data.items() if v}

            if not form_data:
                # If no search criteria provided, return all items
                results = vector_db.get_all_items()
            else:
                results = vector_db.search_by_form_data(form_data, n_results)

        return JSONResponse(
            content={"success": True, "results": results, "count": len(results)}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/generate-refinement-questions")
async def generate_refinement_questions(result_ids: List[str] = Form(...)):
    """Generate quiz questions to help narrow down search results"""
    try:
        print(
            f"🎯 Generating refinement questions for {len(result_ids)} results: {result_ids}"
        )

        # Analyze results to find discriminating features
        analysis = vector_db.analyze_results_for_refinement(result_ids)

        if not analysis["success"]:
            print(f"❌ Analysis failed: {analysis.get('error')}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": analysis.get("error")},
            )

        print(f"✅ Found {len(analysis['discriminators'])} discriminating features")

        # If no discriminators found, return helpful error
        if not analysis["discriminators"]:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Unable to find distinguishing features. The items may have similar metadata (color, brand, condition, etc.)",
                }
            )

        # Generate questions using Watsonx
        questions_result = watsonx_client.generate_refinement_questions(
            analysis["discriminators"], item_category="item"
        )

        if not questions_result["success"]:
            print(f"❌ Question generation failed: {questions_result.get('error')}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": questions_result.get("error")},
            )

        print(f"✅ Generated {len(questions_result['questions'])} questions")

        return JSONResponse(
            content={
                "success": True,
                "questions": questions_result["questions"],
                "discriminators": analysis["discriminators"],
            }
        )

    except Exception as e:
        print(f"❌ Error generating refinement questions: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/refine-search")
async def refine_search(result_ids: List[str] = Form(...), answers: str = Form(...)):
    """Re-rank search results based on quiz answers"""
    try:
        import json

        # Parse answers (expected format: {"field": "value", ...})
        try:
            answers_dict = json.loads(answers)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid answers format"},
            )

        # Get all items
        items = [item for item in vector_db.items if item["id"] in result_ids]

        # Score each item based on answers
        scored_items = []
        for item in items:
            score = 0
            max_score = len(answers_dict)

            # Debug logging
            print(
                f"Scoring {item['id']}: {item['metadata'].get('brand')} {item['metadata'].get('color')} {item['metadata'].get('condition')}"
            )

            for field, expected_value in answers_dict.items():
                # Make field lookup case-insensitive
                field_lower = field.lower()
                item_value = item["metadata"].get(field_lower, "")

                if not item_value:
                    # If lowercase doesn't work, try original case
                    item_value = item["metadata"].get(field, "")

                item_value_lower = str(item_value).lower()
                expected_lower = expected_value.lower()

                print(
                    f"  Comparing {field_lower}: '{item_value_lower}' vs '{expected_lower}'"
                )

                # Exact match gets full point
                if item_value_lower == expected_lower:
                    score += 1
                    print(f"    ✓ Exact match! (+1)")
                # Partial match for certain fields (like material/color variations)
                elif (
                    expected_lower in item_value_lower
                    or item_value_lower in expected_lower
                ):
                    score += 0.5
                    print(f"    ~ Partial match (+0.5)")
                else:
                    print(f"    ✗ No match")

            # Calculate match percentage
            match_percentage = (score / max_score * 100) if max_score > 0 else 0
            print(f"  Final score: {score}/{max_score} = {match_percentage}%\n")

            scored_items.append(
                {
                    "id": item["id"],
                    "description": item["description"],
                    "image_path": item["image_path"],
                    "metadata": item["metadata"],
                    "match_score": score,
                    "match_percentage": round(match_percentage, 1),
                }
            )

        # Sort by score (descending)
        scored_items.sort(key=lambda x: x["match_score"], reverse=True)

        return JSONResponse(
            content={
                "success": True,
                "results": scored_items,
                "count": len(scored_items),
            }
        )

    except Exception as e:
        print(f"Error refining search: {e}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/verify-invoice")
async def verify_invoice(
    invoice: UploadFile = File(...),
    item_id: str = Form(...),
    item_metadata: str = Form(...),
):
    """Verify invoice against top search result item"""
    try:
        import json
        import os
        import tempfile

        print(f"\n📋 Verifying invoice for item {item_id}...")

        # Parse item metadata
        try:
            metadata_dict = json.loads(item_metadata)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid metadata format"},
            )

        # Save uploaded invoice temporarily with correct extension
        file_extension = os.path.splitext(invoice.filename)[1] or ".jpg"
        print(f"Invoice filename: {invoice.filename}, extension: {file_extension}")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            content = await invoice.read()
            temp_file.write(content)
            invoice_path = temp_file.name

        try:
            # Extract invoice data using Watsonx
            print(f"Extracting invoice data from: {invoice_path}")
            invoice_result = watsonx_client.extract_invoice_data(invoice_path)
            print(f"Invoice extraction result: {invoice_result.get('success', False)}")

            if not invoice_result["success"]:
                error_msg = invoice_result.get("error", "Unknown error")
                raw_response = invoice_result.get("raw_response", "")
                print(f"❌ Invoice extraction failed: {error_msg}")
                if raw_response:
                    print(f"Raw response: {raw_response[:200]}...")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": f"Failed to extract invoice data: {error_msg}",
                    },
                )

            invoice_data = invoice_result["data"]
            print(f"Invoice data: {invoice_data}")

            # Compare invoice with item metadata
            discrepancies = []
            match_count = 0
            total_fields = 0

            # Fields to compare (only if present in both)
            comparable_fields = {
                "brand": "Brand",
                "model": "Model",
                "color": "Color",
                "material": "Material",
                "condition": "Condition",
                "category": "Category",
            }

            for field_key, field_name in comparable_fields.items():
                invoice_value = invoice_data.get(field_key)
                item_value = metadata_dict.get(field_key)

                # Only compare if both values exist and are not null
                if invoice_value and item_value and invoice_value != "null":
                    total_fields += 1
                    invoice_lower = str(invoice_value).lower()
                    item_lower = str(item_value).lower()

                    print(
                        f"Comparing {field_name}: invoice='{invoice_lower}' vs item='{item_lower}'"
                    )

                    # Check for match (exact or partial)
                    if invoice_lower == item_lower:
                        match_count += 1
                        print(f"  ✓ Exact match")
                    elif invoice_lower in item_lower or item_lower in invoice_lower:
                        match_count += 0.5
                        print(f"  ~ Partial match")
                    else:
                        discrepancies.append(
                            f"{field_name}: Invoice says '{invoice_value}', but item is '{item_value}'"
                        )
                        print(f"  ✗ Mismatch!")

            # Calculate match percentage
            if total_fields > 0:
                match_percentage = (match_count / total_fields) * 100
            else:
                # If no comparable fields, consider it uncertain
                match_percentage = 50.0

            print(f"\nMatch score: {match_count}/{total_fields} = {match_percentage}%")
            print(f"Discrepancies: {discrepancies}")

            # Determine match status
            if match_percentage >= 70:
                match_status = "match"
                confidence = "High" if match_percentage >= 85 else "Medium"
            elif match_percentage >= 40:
                match_status = "partial"
                confidence = "Low"
            else:
                match_status = "mismatch"
                confidence = "Very Low"

            return JSONResponse(
                content={
                    "success": True,
                    "match_status": match_status,
                    "match_percentage": round(match_percentage, 1),
                    "confidence": confidence,
                    "discrepancies": discrepancies,
                    "invoice_data": invoice_data,
                }
            )

        finally:
            # Clean up temporary file
            if os.path.exists(invoice_path):
                os.remove(invoice_path)

    except Exception as e:
        print(f"❌ Error verifying invoice: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.get("/description", response_class=HTMLResponse)
async def description_page():
    """Serve the description generation page"""
    with open("frontend/templates/description.html", "r") as f:
        return f.read()


@router.get("/search", response_class=HTMLResponse)
async def search_page():
    """Serve the search page"""
    with open("frontend/templates/search.html", "r") as f:
        return f.read()
