from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os
import uuid
import shutil
from typing import Optional, Dict, Any
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
            nl_result = watsonx_client.process_natural_language_search(natural_query)

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
