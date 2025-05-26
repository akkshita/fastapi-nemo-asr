#!/usr/bin/env python3
"""
Generate a simple test audio file for testing the ASR application.
This creates a sine wave tone that can be used to test the audio processing pipeline.
"""

import os
import numpy as np
import soundfile as sf
import argparse

def generate_sine_wave(freq=440, duration=5, sample_rate=16000):
    """Generate a sine wave at the given frequency, duration, and sample rate."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return sine_wave

def main():
    parser = argparse.ArgumentParser(description="Generate a test audio file")
    parser.add_argument("--output", default="sample.wav", help="Output file path")
    parser.add_argument("--freq", type=int, default=440, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    args = parser.parse_args()
    
    # Generate the sine wave
    audio = generate_sine_wave(args.freq, args.duration, args.sample_rate)
    
    # Save the audio file
    sf.write(args.output, audio, args.sample_rate)
    
    print(f"Generated test audio file: {args.output}")
    print(f"  Frequency: {args.freq} Hz")
    print(f"  Duration: {args.duration} seconds")
    print(f"  Sample rate: {args.sample_rate} Hz")

if __name__ == "__main__":
    main()
