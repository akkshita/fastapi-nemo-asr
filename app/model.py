import os
import logging
import tempfile
import asyncio
import librosa
import soundfile as sf
import numpy as np
from typing import Dict, Any, Optional
import onnxruntime as ort
from utils.model_converter import download_and_convert_to_onnx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRModel:
    """Class for handling ASR model operations."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ASR model.
        
        Args:
            model_path: Path to the ONNX model file. If None, will use the default path.
        """
        # Set model paths
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_path = model_path or os.path.join(self.model_dir, "stt_hi_conformer_ctc_medium.onnx")
        self.vocab_path = os.path.join(self.model_dir, "vocab.txt")
        
        # Check if model exists, if not, download and convert
        if not os.path.exists(self.model_path) or not os.path.exists(self.vocab_path):
            logger.info("Model or vocabulary file not found. Downloading and converting...")
            download_and_convert_to_onnx(self.model_dir)
        
        # Load the model
        self._load_model()
        
        # Load vocabulary
        self.vocab = self._load_vocabulary()
        
        # Set audio parameters
        self.sample_rate = 16000  # 16 kHz
    
    def _load_model(self):
        """Load the ONNX model."""
        try:
            # Create an ONNX Runtime session
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_vocabulary(self):
        """Load the vocabulary file."""
        try:
            with open(self.vocab_path, 'r') as f:
                vocab = [line.strip() for line in f]
            logger.info(f"Vocabulary loaded with {len(vocab)} tokens")
            return vocab
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            raise
    
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
        Preprocess the audio file for the model.
        
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
            
            # Convert to float32
            audio = audio.astype(np.float32)
            
            # Reshape for model input
            audio = np.expand_dims(audio, axis=0)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def _decode_output(self, output: np.ndarray) -> str:
        """
        Decode the model output to text.
        
        Args:
            output: Model output logits
            
        Returns:
            Decoded text
        """
        try:
            # Get the most likely token indices
            token_indices = np.argmax(output, axis=-1)[0]
            
            # CTC decoding (merge repeated tokens and remove blanks)
            previous_token = None
            decoded_text = []
            
            for token_idx in token_indices:
                # Skip if token is blank (usually index 0)
                if token_idx == 0:
                    continue
                
                # Skip if token is the same as the previous one (CTC merge)
                if token_idx != previous_token:
                    decoded_text.append(self.vocab[token_idx])
                
                previous_token = token_idx
            
            # Join the tokens to form the final text
            return ''.join(decoded_text)
        except Exception as e:
            logger.error(f"Error decoding output: {str(e)}")
            raise
    
    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Run inference in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self.session.run([self.output_name], {self.input_name: audio})[0]
            )
            
            # Decode the output
            transcription = self._decode_output(output)
            
            return transcription
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
