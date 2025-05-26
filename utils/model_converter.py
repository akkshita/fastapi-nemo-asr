import os
import logging
import subprocess
import tempfile
import shutil
import torch
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NeMo model URL
NEMO_MODEL_URL = "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files"

def download_and_convert_to_onnx(output_dir: str, model_name: str = "stt_hi_conformer_ctc_medium") -> None:
    """
    Download the NeMo model and convert it to ONNX format.
    
    Args:
        output_dir: Directory to save the model
        model_name: Name of the model
    """
    try:
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the model
            logger.info(f"Downloading NeMo model: {model_name}")
            download_model(temp_dir, model_name)
            
            # Convert to ONNX
            logger.info("Converting model to ONNX format")
            convert_to_onnx(temp_dir, output_dir, model_name)
            
            # Extract vocabulary file
            extract_vocabulary(temp_dir, output_dir)
            
            logger.info(f"Model successfully converted and saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during model download and conversion: {str(e)}")
        raise

def download_model(temp_dir: str, model_name: str) -> None:
    """
    Download the NeMo model using the NVIDIA NGC CLI.
    
    Args:
        temp_dir: Temporary directory to download the model
        model_name: Name of the model
    """
    try:
        # Check if NGC CLI is installed
        try:
            subprocess.run(["ngc", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.info("NGC CLI not found. Installing...")
            install_ngc_cli()
        
        # Download the model
        download_cmd = [
            "ngc", "registry", "model", "download-version",
            f"nvidia/nemo/{model_name}:1.20.0",
            "--dest", temp_dir
        ]
        
        subprocess.run(download_cmd, check=True)
        logger.info(f"Model {model_name} downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def install_ngc_cli() -> None:
    """Install the NVIDIA NGC CLI."""
    try:
        # Download and install NGC CLI
        subprocess.run([
            "wget", "-O", "/tmp/ngccli_linux.zip",
            "https://ngc.nvidia.com/downloads/ngccli_linux.zip"
        ], check=True)
        
        subprocess.run([
            "unzip", "-o", "/tmp/ngccli_linux.zip", "-d", "/tmp"
        ], check=True)
        
        os.environ["PATH"] += os.pathsep + "/tmp/ngc-cli"
        logger.info("NGC CLI installed successfully")
    except Exception as e:
        logger.error(f"Error installing NGC CLI: {str(e)}")
        raise

def convert_to_onnx(input_dir: str, output_dir: str, model_name: str) -> None:
    """
    Convert the NeMo model to ONNX format.
    
    Args:
        input_dir: Directory containing the downloaded model
        output_dir: Directory to save the converted model
        model_name: Name of the model
    """
    try:
        # Find the .nemo file
        nemo_file = None
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".nemo"):
                    nemo_file = os.path.join(root, file)
                    break
            if nemo_file:
                break
        
        if not nemo_file:
            raise FileNotFoundError("No .nemo file found in the downloaded model")
        
        # Create a Python script for conversion
        script_path = os.path.join(output_dir, "convert_to_onnx.py")
        with open(script_path, "w") as f:
            f.write("""
import os
import torch
from nemo.collections.asr.models import EncDecCTCModel

# Load the model
model = EncDecCTCModel.restore_from("{nemo_file}")

# Export to ONNX
model.export("{output_file}", onnx_opset_version=13)

# Save vocabulary
with open("{vocab_file}", "w") as f:
    for token in model.decoder.vocabulary:
        f.write(token + "\\n")
""".format(
                nemo_file=nemo_file,
                output_file=os.path.join(output_dir, f"{model_name}.onnx"),
                vocab_file=os.path.join(output_dir, "vocab.txt")
            ))
        
        # Run the conversion script
        subprocess.run([
            "python", script_path
        ], check=True)
        
        # Clean up
        os.remove(script_path)
        
        logger.info(f"Model converted to ONNX and saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        raise

def extract_vocabulary(input_dir: str, output_dir: str) -> None:
    """
    Extract vocabulary from the model if not already done.
    
    Args:
        input_dir: Directory containing the downloaded model
        output_dir: Directory to save the vocabulary
    """
    # Check if vocabulary file already exists
    vocab_file = os.path.join(output_dir, "vocab.txt")
    if os.path.exists(vocab_file):
        logger.info(f"Vocabulary file already exists at {vocab_file}")
        return
    
    try:
        # Find the vocabulary file in the downloaded model
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file == "vocab.txt" or file == "vocabulary.txt":
                    src_vocab = os.path.join(root, file)
                    shutil.copy(src_vocab, vocab_file)
                    logger.info(f"Vocabulary file copied to {vocab_file}")
                    return
        
        logger.warning("No vocabulary file found in the downloaded model")
    except Exception as e:
        logger.error(f"Error extracting vocabulary: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the converter
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_and_convert_to_onnx(os.path.join(output_dir, "models"))
