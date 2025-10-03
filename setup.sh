#!/bin/bash

echo "🏢 Setting up Overgoods AI System..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p static
mkdir -p chroma_db

echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set your Watsonx credentials in a .env file or export them:"
echo "   export WATSONX_API_KEY='your_api_key'"
echo "   export WATSONX_PROJECT_ID='your_project_id'"
echo "3. Run the server: python run.py"
echo "4. Open http://localhost:8000 in your browser"

