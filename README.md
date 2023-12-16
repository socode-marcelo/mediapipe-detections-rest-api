 # FastAPI Application for Image Detection

This is a FastAPI application uses the Mediapipe library to perform face and object detection on images. It provides endpoints to detect faces and objects from file, base64 encoded image or numpy array.


1. A Dockerfile for building the image from scratch using Python 3.9, Pillow (for handling images), NumPy (for array operations), and mediapipe (a library for computer vision tasks).
2. An endpoint `/` that returns a simple health check message to ensure the server is running smoothly.
3. Endpoints for detecting faces in an image from different sources:
	* A file upload endpoint (`/detect_face_from_file`) that accepts a single file and returns the detected face coordinates as JSON data.
	* An array input endpoint (`/detect_face_from_array`) that takes a list of integers representing RGB values for an image and returns the same JSON data.
	* A base64-encoded string input endpoint (`/detect_face_from_base64`) that accepts a base64-encoded image file and returns the same JSON data.
4. Endpoints for detecting objects in an image from different sources:
	* Similar to the face detection endpoints, but with object detector options set to return up to 5 results with a score threshold of 0.5.

The code uses Pydantic models and HTTPExceptions for request validation and error handling. The `detect_faces` and `detect_objects` functions use MediaPipe's FaceDetector and ObjectDetector classes, respectively, to perform the actual detection tasks on NumPy arrays or image files.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<username>/image-detection-fastapi.git
cd image-detection-fastapi
```
2. Create a virtual environment and activate it:
```bash
python3 -m venv env
source env/bin/activate (on Linux/macOS)
env\Scripts\activate (on Windows)
```
3. Install the required packages:
```bash
pip install --no-cache-dir mediapipe fastapi uvicorn sentencepiece python-multipart
```
4. Run the application using uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
## API Endpoints

### Health Check

#### `GET /`

Check if the server is running smoothly. Returns a JSON object with status information.

**Example response:**
```json
{
    "status": "Server is running smoothly!"
}
```
---

### Face Detection Endpoints

#### `POST /detect_face_from_file`

Detect faces in an image from a file. Returns the detection results as JSON object.

**Request body:** File (image)


#### `POST /detect_face_from_array`

Detect faces in a numpy array image. Returns the detection results as JSON object.

**Request body:** List[int] (numpy array)


#### `POST /detect_face_from_base64`

Detect faces in an image from a base64 encoded string. Returns the detection results as JSON object.

**Request body:** DetectionRequest (image\_path: str)

### Example response:

```json
{
    "face_detector_result": [
        {
            "bounding_box": [x1, y1, x2, y2],
            "score": 0.9876543
        },
        ...
    ]
}
```
---

### Object Detection Endpoints

#### `POST /detect_objects_from_file`

Detect objects in an image from a file. Returns the detection results as JSON object.

**Request body:** File (image)


#### `POST /detect_objects_from_array`

Detect objects in a numpy array image. Returns the detection results as JSON object.

**Request body:** List[int] (numpy array)



#### `POST /detect_objects_from_base64`

Detect objects in an image from a base64 encoded string. Returns the detection results as JSON object.

**Request body:** DetectionRequest (image\_path: str)

### Example response:
```json
{
    "object_detector_result": [
        {
            "bounding_box": [x1, y1, x2, y2],
            "label": "person",
            "score": 0.9876543
        },
        ...
    ]
}
```