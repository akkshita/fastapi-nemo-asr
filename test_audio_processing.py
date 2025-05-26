#!/usr/bin/env python3
"""
Test script for audio processing functionality.
This script tests the audio loading and preprocessing without requiring the NeMo model.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def process_audio(audio_path, sample_rate=16000):
    """
    Process an audio file and return basic information and features.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate
        
    Returns:
        Dictionary with audio information and features
    """
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Get duration
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Normalize audio
    audio_normalized = audio / np.max(np.abs(audio))
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13)
    
    # Generate spectrogram
    spec = np.abs(librosa.stft(audio_normalized))
    
    return {
        "filename": os.path.basename(audio_path),
        "duration": duration,
        "sample_rate": sr,
        "num_samples": len(audio),
        "mfccs": mfccs,
        "spectrogram": spec
    }

def save_audio_visualization(audio_info, output_dir="."):
    """
    Save visualizations of the audio features.
    
    Args:
        audio_info: Dictionary with audio information and features
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(audio_info["spectrogram"], ref=np.max),
        sr=audio_info["sample_rate"],
        y_axis='log',
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {audio_info["filename"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(audio_info['filename'])[0]}_spectrogram.png"))
    plt.close()
    
    # Plot MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        audio_info["mfccs"],
        sr=audio_info["sample_rate"],
        x_axis='time'
    )
    plt.colorbar()
    plt.title(f'MFCC - {audio_info["filename"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(audio_info['filename'])[0]}_mfcc.png"))
    plt.close()

def main():
    # Process the test audio file
    audio_path = "test_sample.wav"
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found")
        print("Please run 'python utils/generate_test_audio.py' first")
        return
    
    # Process the audio
    audio_info = process_audio(audio_path)
    
    # Print audio information
    print("\nAudio Information:")
    print(f"Filename: {audio_info['filename']}")
    print(f"Duration: {audio_info['duration']:.2f} seconds")
    print(f"Sample Rate: {audio_info['sample_rate']} Hz")
    print(f"Number of Samples: {audio_info['num_samples']}")
    
    # Create output directory for visualizations
    output_dir = "static"
    
    # Save visualizations
    try:
        save_audio_visualization(audio_info, output_dir)
        print(f"\nVisualizations saved to {output_dir}/")
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
    
    print("\nAudio processing test completed successfully!")

if __name__ == "__main__":
    main()
