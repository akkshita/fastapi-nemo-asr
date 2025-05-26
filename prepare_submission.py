#!/usr/bin/env python3
"""
Submission preparation script for the ASR application assignment.
This script checks the project structure, verifies key files, and creates a summary.
"""

import os
import sys
import shutil
from pathlib import Path
import datetime

def check_file_exists(file_path, required=True):
    """Check if a file exists and print its status."""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌" if required else "⚠️"
    print(f"{status} {file_path}")
    return exists

def count_lines_of_code(file_path):
    """Count the number of lines in a file, excluding blank lines and comments."""
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Count non-blank, non-comment lines
    count = 0
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
            count += 1
    
    return count

def create_submission_summary():
    """Create a summary of the submission."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create summary file
    summary_path = os.path.join(project_dir, "SUBMISSION_SUMMARY.md")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# ASR Application Submission Summary\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Project structure
        f.write("## Project Structure\n\n")
        f.write("```\n")
        
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(project_dir):
            # Skip __pycache__ directories
            if "__pycache__" in root:
                continue
                
            level = root.replace(project_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            subindent = ' ' * 4 * (level + 1)
            
            rel_root = os.path.relpath(root, project_dir)
            if rel_root == '.':
                f.write(f"{os.path.basename(project_dir)}/\n")
            else:
                f.write(f"{indent}{os.path.basename(root)}/\n")
            
            for file in sorted(files):
                # Skip this script and the summary file
                if file == os.path.basename(__file__) or file == os.path.basename(summary_path):
                    continue
                    
                file_path = os.path.join(root, file)
                all_files.append(file_path)
                f.write(f"{subindent}{file}\n")
        
        f.write("```\n\n")
        
        # Implementation status
        f.write("## Implementation Status\n\n")
        
        # Check key files
        f.write("### Key Files\n\n")
        
        key_files = [
            ("Dockerfile", True),
            ("requirements.txt", True),
            ("README.md", True),
            ("Description.md", True),
            ("app/main.py", True),
            ("app/model.py", True),
            ("utils/model_converter.py", True),
            ("simple_api.py", False),
            ("test_audio_processing.py", False),
            ("utils/generate_test_audio.py", False),
            ("test_sample.wav", False)
        ]
        
        for file_path, required in key_files:
            full_path = os.path.join(project_dir, file_path)
            check_file_exists(full_path, required)
        
        f.write("\n")
        
        # Code statistics
        f.write("### Code Statistics\n\n")
        
        total_lines = 0
        for file_path in all_files:
            if file_path.endswith('.py'):
                lines = count_lines_of_code(file_path)
                total_lines += lines
                rel_path = os.path.relpath(file_path, project_dir)
                f.write(f"- {rel_path}: {lines} lines\n")
        
        f.write(f"\nTotal lines of Python code: {total_lines}\n\n")
        
        # Requirements
        f.write("## Requirements Checklist\n\n")
        
        requirements = [
            ("Model Preparation", "Use the ASR model from NVIDIA NeMo and optimize for inference using ONNX"),
            ("FastAPI Application", "Create a FastAPI server with a /transcribe endpoint"),
            ("Containerization", "Create a Dockerfile to containerize the application"),
            ("Documentation", "Provide README.md and Description.md with required information"),
            ("Audio Processing", "Handle audio files of 5-10 seconds, sampled at 16kHz")
        ]
        
        for req, desc in requirements:
            f.write(f"- [ ] {req}: {desc}\n")
        
        f.write("\n")
        f.write("Please check the boxes for completed requirements before submission.\n\n")
        
        # Submission instructions
        f.write("## Submission Instructions\n\n")
        f.write("1. Verify that all required files are present and complete\n")
        f.write("2. Check the requirements checklist and mark completed items\n")
        f.write("3. Submit the entire project directory as a compressed file (zip or tar.gz)\n")
        f.write("4. Ensure the submission includes any test audio clips if available\n")
        
    print(f"\nSubmission summary created at: {summary_path}")
    return summary_path

def main():
    """Main function to prepare the submission."""
    print("Preparing submission for ASR application assignment...\n")
    
    # Check project structure
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Project directory: {project_dir}\n")
    
    print("Checking key directories...")
    directories = ["app", "utils", "static"]
    for directory in directories:
        dir_path = os.path.join(project_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created missing directory: {dir_path}")
    
    # Create submission summary
    summary_path = create_submission_summary()
    
    print("\nSubmission preparation complete!")
    print("Please review the summary file and make any necessary adjustments before submitting.")

if __name__ == "__main__":
    main()
