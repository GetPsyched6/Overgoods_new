# AI Vision System

A modern AI-powered computer vision system for automated item analysis and intelligent search.

## Features

- **AI Product Analysis**: Upload images to get detailed AI-generated descriptions with confidence scores
- **Smart Search**: Search through items using natural language, form filters, or image similarity
- **Secondary Characteristics**: Advanced analysis detecting primary and secondary colors/materials with percentages
- **Modern UI**: Clean dark/light mode interface with smooth animations and typewriter style text effects

## Quick Start

1. **Setup Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:

   ```bash
   cp sample.env .env
   # Edit .env with your Watsonx credentials
   ```

3. **Add Your Images**:

   - Place your item images in the `data/assets/` folder
   - Supported formats: JPG, PNG, GIF, WebP

4. **Generate AI Descriptions** (Optional):

   ```bash
   python regenerate.py
   ```

   This will analyze all images in `data/assets/` and generate AI descriptions for better search results.

5. **Run the Application**:

   ```bash
   python run.py
   ```

6. **Access the System**:
   - Open <http://localhost:8000> in your browser
   - Use the Product Analysis page to analyze individual items
   - Use the Smart Search page to find similar items

## Project Structure

```bash
├── app/                    # Main application package
│   ├── api/               # FastAPI routes and endpoints
│   ├── core/              # Configuration and core utilities
│   ├── services/          # Business logic and external services
│   ├── scripts/           # Utility scripts
│   └── models/            # Data models (future expansion)
├── frontend/              # Web interface
│   └── templates/         # HTML templates
├── data/                  # Application data
│   └── assets/           # Item images for analysis
├── tests/                 # Test files
├── chroma_db/            # Vector database storage
├── uploads/              # Temporary upload storage
├── run.py                # Application entry point
├── regenerate.py         # Convenience script for AI descriptions
└── sample.env            # Environment template
```

## Technology Stack

- **Backend**: FastAPI, Python 3.13+
- **AI/ML**: IBM Watsonx (Llama-3-2-90B-vision-instruct)
- **Frontend**: Modern HTML5, CSS3, JavaScript
- **Storage**: File-based vector database with JSON persistence

## Requirements

- Python 3.13+
- IBM Watsonx API access
- 10MB+ available storage for images and cache

### Shared Google Drive Folder (upload images of boxes here)

<https://drive.google.com/drive/folders/1Okq8hk7-HQPaypJhPSxpSQIGxehjY6fB?usp=sharing>

### Shared Google Doc (ideas)

<https://docs.google.com/document/d/1HbyCEStxCLpo2de2UuN6kggZ4TjA56ZN7fImw_dLSwo/edit?usp=sharing>

## License

Private project - All rights reserved.
