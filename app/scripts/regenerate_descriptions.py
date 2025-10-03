#!/usr/bin/env python3
"""
Script to regenerate AI descriptions for all assets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.vector_service import SimpleVectorDatabase

def main():
    print("🤖 Regenerating AI descriptions for all assets...")
    
    # Initialize the vector database
    vector_db = SimpleVectorDatabase()
    
    # Generate AI descriptions
    result = vector_db.generate_ai_descriptions()
    
    if result["success"]:
        print(f"✅ Successfully updated {result['updated']} items with AI descriptions!")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
