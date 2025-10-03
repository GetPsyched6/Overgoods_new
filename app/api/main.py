from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app = FastAPI(title="AI Vision System", version="1.0.0")

# Include API routes
app.include_router(router)

# Mount static files
app.mount("/assets", StaticFiles(directory="data/assets"), name="assets")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")