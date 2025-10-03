#!/usr/bin/env python3
"""
AI Vision System - Main Entry Point

A modern AI-powered computer vision system for automated item analysis and search.
"""

if __name__ == "__main__":
    import uvicorn
    from app.api.main import app

    uvicorn.run(app, host="0.0.0.0", port=8000)
