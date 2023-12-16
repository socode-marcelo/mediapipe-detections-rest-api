# Use the base image from Hugging Face
FROM mediapipe

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
#COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN wget -q -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
RUN wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite

# Install the required dependencies
RUN pip install --no-cache-dir mediapipe fastapi uvicorn sentencepiece python-multipart

# Copy the main.py file
COPY main.py .

# Expose the port on which the API will run
EXPOSE 8000

# Check if the server is responding
HEALTHCHECK --interval=60s --timeout=4s CMD curl --fail http://localhost:8000/openapi.json || exit 1

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]