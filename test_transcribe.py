#!/usr/bin/env python3
"""
Test script for the ASR model functionality.
This script allows you to test the transcription functionality without running the full FastAPI server.
"""

import os
import sys
import asyncio
import argparse
from app.model import ASRModel

async def main():
    parser = argparse.ArgumentParser(description="Test the ASR model transcription")
    parser.add_argument("audio_file", help="Path to the audio file (.wav) to transcribe")
    args = parser.parse_args()
    
    # Check if the audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)
    
    # Check if the audio file is a WAV file
    if not args.audio_file.lower().endswith('.wav'):
        print("Error: Only .wav files are supported")
        sys.exit(1)
    
    print("Initializing ASR model...")
    model = ASRModel()
    
    # Get audio duration
    duration = model.get_audio_duration(args.audio_file)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Transcribe the audio
    print("Transcribing audio...")
    transcription = await model.transcribe(args.audio_file)
    
    print("\nResults:")
    print(f"Filename: {os.path.basename(args.audio_file)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    asyncio.run(main())
