import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any

from app.model import ASRModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NeMo ASR API",
    description="API for Hindi speech recognition using NVIDIA NeMo",
    version="1.0.0"
)

# Initialize the ASR model
asr_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ASR model on startup."""
    global asr_model
    try:
        asr_model = ASRModel()
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "NeMo ASR API is running"}

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file using the ASR model.
    
    Args:
        file: Audio file (.wav) to transcribe
        
    Returns:
        JSON response with transcribed text
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
        duration = asr_model.get_audio_duration(temp_file_path)
        if duration < 1 or duration > 15:  # Slightly wider range for flexibility
            os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Audio duration ({duration:.2f}s) must be between 1 and 15 seconds"
            )
        
        # Transcribe the audio
        transcription = await asr_model.transcribe(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return {
            "filename": file.filename,
            "duration": f"{duration:.2f}s",
            "transcription": transcription
        }
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals():
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
