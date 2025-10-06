@echo off
echo Setting up Overgoods AI System...

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo Creating directories...
if not exist "uploads" mkdir uploads
if not exist "static" mkdir static
if not exist "chroma_db" mkdir chroma_db

echo Setup complete!
echo.
echo To run the application:
echo 1. Activate: venv\Scripts\activate.bat
echo 2. Set environment variables in .env file
echo 3. Run: python run.py