#!/usr/bin/env python3
"""
Simple FastAPI application for audio processing.
This demonstrates the API structure without requiring the full NeMo model.
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Processing API",
    description="API for audio processing and analysis",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class SimpleAudioProcessor:
    """Class for handling audio processing operations."""
    
    def __init__(self):
        """Initialize the audio processor."""
        self.sample_rate = 16000  # 16 kHz
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration of the audio in seconds
        """
        try:
            return librosa.get_duration(path=audio_path)
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Ensure correct sample rate
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            
            # Extract zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            return {
                "mfccs_mean": np.mean(mfccs, axis=1).tolist(),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate))
            }
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    async def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Process an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get audio duration
            duration = self.get_audio_duration(audio_path)
            
            # Preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Run feature extraction in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(
                None,
                lambda: self.extract_features(audio)
            )
            
            # Mock transcription (since we don't have the actual NeMo model)
            mock_transcription = "This is a simulated transcription. In the full implementation, this would be the output from the NeMo ASR model."
            
            return {
                "duration": duration,
                "sample_rate": self.sample_rate,
                "features": features,
                "transcription": mock_transcription
            }
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise

# Initialize the audio processor
audio_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the audio processor on startup."""
    global audio_processor
    try:
        audio_processor = SimpleAudioProcessor()
        logger.info("Audio processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio processor: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Audio Processing API is running"}

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Process an audio file.
    
    Args:
        file: Audio file (.wav) to process
        
    Returns:
        JSON response with processing results
    """
    # Validate file type
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    # Save the uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
        
        # Validate audio duration
        duration = audio_processor.get_audio_duration(temp_file_path)
        if duration < 1 or duration > 15:  # Slightly wider range for flexibility
            os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Audio duration ({duration:.2f}s) must be between 1 and 15 seconds"
            )
        
        # Process the audio
        results = await audio_processor.process(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return {
            "filename": file.filename,
            "duration": f"{results['duration']:.2f}s",
            "sample_rate": f"{results['sample_rate']} Hz",
            "features": results['features'],
            "transcription": results['transcription']
        }
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals():
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)
