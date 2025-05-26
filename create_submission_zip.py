#!/usr/bin/env python3
"""
Script to create a submission zip file for the ASR application assignment.
"""

import os
import sys
import zipfile
import datetime
from pathlib import Path

def create_submission_zip():
    """Create a zip file containing all project files for submission."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(project_dir)
    
    # Generate timestamp for the zip filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{project_name}_submission_{timestamp}.zip"
    zip_path = os.path.join(os.path.dirname(project_dir), zip_filename)
    
    # Files and directories to exclude
    exclude = [
        "__pycache__",
        ".git",
        ".gitignore",
        ".DS_Store",
        "venv",
        os.path.basename(__file__),  # Exclude this script
        "create_submission_zip.py"
    ]
    
    print(f"Creating submission zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files and directories
        for root, dirs, files in os.walk(project_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude]
            
            # Get the relative path from the project directory
            rel_path = os.path.relpath(root, os.path.dirname(project_dir))
            
            # Add files
            for file in files:
                # Skip excluded files
                if file in exclude:
                    continue
                
                file_path = os.path.join(root, file)
                # Get the path relative to the parent of the project directory
                arc_path = os.path.join(rel_path, file)
                
                print(f"Adding: {arc_path}")
                zipf.write(file_path, arc_path)
    
    print(f"\nSubmission zip file created successfully: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path) / (1024 * 1024):.2f} MB")
    
    return zip_path

def main():
    """Main function to create the submission zip file."""
    print("Preparing submission zip file for ASR application assignment...\n")
    
    # Create submission zip
    zip_path = create_submission_zip()
    
    print("\nSubmission preparation complete!")
    print("Please submit this zip file by the deadline: May 26, 2025, 1:00 p.m.")

if __name__ == "__main__":
    main()
