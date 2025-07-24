#!/usr/bin/env python3
"""
Fix missing dataset splits by creating empty JSON files
"""
import json
from pathlib import Path

def create_empty_json_files():
    """Create empty JSON files for missing splits"""
    
    # Missing files that need to be created
    missing_files = [
        "data/surreal_3d/train/run1.json",
        "data/surreal_3d/train/run2.json", 
        "data/surreal_3d/val/run0.json",
        "data/surreal_3d/test/run0.json"
    ]
    
    for file_path in missing_files:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty JSON file
        with open(path, 'w') as f:
            json.dump([], f)
        
        print(f"✅ Created empty file: {file_path}")

if __name__ == "__main__":
    print("🔧 Fixing missing dataset splits...")
    create_empty_json_files()
    print("✅ Done!") 