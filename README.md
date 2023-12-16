 # FastAPI Application for Image Detection

This application uses the Mediapipe library to perform face and object detection on images. The application provides endpoints to detect faces and objects from file, base64 encoded image or numpy array.

## Prerequisites

* Python 3.7+
* FastAPI
* Uvicorn (ASGI server)
* Mediapipe
* Numpy
* Pillow

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
pip install -r requirements.txt
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

**Example response:**
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
#### `POST /detect_face_from_array`

Detect faces in a numpy array image. Returns the detection results as JSON object.

**Request body:** List[int] (numpy array)

**Example response:**
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
#### `POST /detect_face_from_base64`

Detect faces in an image from a base64 encoded string. Returns the detection results as JSON object.

**Request body:** DetectionRequest (image\_path: str)

**Example response:**
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

**Example response:**
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
#### `POST /detect_objects_from_array`

Detect objects in a numpy array image. Returns the detection results as JSON object.

**Request body:** List[int] (numpy array)

**Example response:**
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
#### `POST /detect_objects_from_base64`

Detect objects in an image from a base64 encoded string. Returns the detection results as JSON object.
