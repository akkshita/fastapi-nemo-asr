# FastAPI-based ASR Application Using NVIDIA NeMo

This application serves an Automatic Speech Recognition (ASR) model built using NVIDIA NeMo, optimized for inference using ONNX. The model can transcribe Hindi audio clips of 5-10 seconds.

## Features

- FastAPI server with `/transcribe` endpoint for audio transcription
- ONNX-optimized NVIDIA NeMo ASR model for Hindi
- Docker containerization for easy deployment
- Asynchronous inference pipeline for better performance

## Project Structure

```
nemo_asr_app/
├── app/                  # Main application code
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── model.py          # ASR model handling
├── utils/                # Utility scripts
│   ├── __init__.py
│   ├── generate_test_audio.py  # Generate test audio files
│   └── model_converter.py      # Convert NeMo model to ONNX
├── static/               # Static files and visualizations
├── models/               # Directory for storing models (created at runtime)
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── Description.md        # Implementation details and challenges
├── simple_api.py         # Simplified API for testing
└── test_audio_processing.py  # Test script for audio processing
```

## Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Test Audio

```bash
python utils/generate_test_audio.py --output test_sample.wav --duration 5
```

### 4. Test Audio Processing

```bash
python test_audio_processing.py
```

### 5. Run the Simple API

```bash
python simple_api.py
```

The API will be accessible at http://localhost:8000.

## Testing the Simple API

### Using curl

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_sample.wav"
```

### Using the Swagger UI

Open http://localhost:8000/docs in your browser to access the interactive API documentation.

## Full Implementation with Docker

### Build the Docker Image

```bash
docker build -t nemo-asr-app .
```

### Run the Container

```bash
docker run -p 8000:8000 nemo-asr-app
```

The application will be accessible at http://localhost:8000.

**Note:** On first run, the application will download and convert the NeMo model to ONNX format, which may take some time depending on your internet connection and hardware.

## API Usage

### Transcribe Endpoint

**Endpoint:** `POST /transcribe`

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV format, 16kHz sample rate)

**Sample curl request:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_sample.wav"
```

**Sample Postman request:**
1. Set the request type to `POST`
2. Enter the URL: `http://localhost:8000/transcribe`
3. Go to the `Body` tab
4. Select `form-data`
5. Add a key named `file` of type `File`
6. Upload your WAV file
7. Click `Send`

**Response:**
```json
{
  "filename": "test_sample.wav",
  "duration": "5.00s",
  "sample_rate": "16000 Hz",
  "features": {
    "mfccs_mean": [-403.69, 41.83, 26.27, 9.15, -9.42, -25.77, -36.56, -39.83, -35.13, -23.62, -7.89, 8.76, 22.97],
    "spectral_centroid_mean": 443.18,
    "zero_crossing_rate_mean": 0.054
  },
  "transcription": "नमस्ते यह एक परीक्षण वाक्य है"
}
```

## Design Considerations

1. **Model Optimization**: The model is converted to ONNX format for faster inference and better portability.
2. **Asynchronous Processing**: The API uses FastAPI's async capabilities and runs model inference in a separate thread to avoid blocking the main event loop.
3. **Input Validation**: The API validates audio files for format and duration before processing.
4. **Error Handling**: Comprehensive error handling with appropriate HTTP status codes and error messages.
5. **Docker Optimization**: The Dockerfile is optimized for smaller image size using a slim base image and multi-stage builds.
6. **Resource Management**: Temporary files are properly cleaned up after processing.
