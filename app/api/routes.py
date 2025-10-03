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


@router.post("/api/reprocess-descriptions")
async def reprocess_descriptions():
    """Reprocess all items with AI descriptions"""
    try:
        result = vector_db.generate_ai_descriptions()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/api/generate-description")
async def generate_description(file: UploadFile = File(...)):
    """Generate product description from uploaded image"""
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file)

        # Generate description using Watsonx
        result = watsonx_client.generate_product_description(file_path)

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
            # Image-based search
            file_path = save_uploaded_file(file)

            # Generate description for the uploaded image
            embedding_result = watsonx_client.generate_search_embedding(file_path)

            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass

            if embedding_result["success"]:
                # Search using the generated description
                results = vector_db.search_by_text(
                    embedding_result["description"], n_results
                )
                print(f"Image search query: {embedding_result['description'][:100]}...")
            else:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": embedding_result["error"]},
                )
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
