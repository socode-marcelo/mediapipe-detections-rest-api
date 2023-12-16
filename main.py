from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import base64
from pydantic import BaseModel

app = FastAPI()

FaceBaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceVisionRunningMode = mp.tasks.vision.RunningMode
FaceModelPath = '/app/detector.tflite'

ObjectBaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
ObjectVisionRunningMode = mp.tasks.vision.RunningMode
ObjectModelPath = '/app/efficientdet.tflite'

class DetectionRequest(BaseModel): # Pydantic model for request body validation
    image_path: str

def detect_faces(numpy_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    options = FaceDetectorOptions(
        base_options=FaceBaseOptions(model_asset_path=FaceModelPath),
        running_mode=FaceVisionRunningMode.IMAGE)
    with FaceDetector.create_from_options(options) as detector:
        return detector.detect(mp_image)

def detect_objects(numpy_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    options = ObjectDetectorOptions(
        base_options=ObjectBaseOptions(model_asset_path=ObjectModelPath),
        max_results=5,
        score_threshold=0.5,
        running_mode=ObjectVisionRunningMode.IMAGE)
    with ObjectDetector.create_from_options(options) as detector:
        return detector.detect(mp_image)

@app.get("/")
def health_check():
    return {"status": "Server is running smoothly!"}

@app.post("/detect_face_from_file")
async def detect_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        numpy_image = np.array(image)

        return {"face_detector_result": detect_faces(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/detect_face_from_array")
async def detect_from_array(numpy_image: List[int]):
    try:
        numpy_image = np.array(numpy_image)

        return {"face_detector_result": detect_faces(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/detect_face_from_base64")
async def detect_from_base64(request: DetectionRequest):
    try:
        decoded_image = base64.b64decode(request.image_path.split(',')[1])
        image = Image.open(io.BytesIO(decoded_image)).convert('RGB')
        numpy_image = np.array(image)

        return {"face_detector_result": detect_faces(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/detect_objects_from_file")
async def detect_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        numpy_image = np.array(image)

        return {"object_detector_result": detect_objects(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/detect_objects_from_array")
async def detect_from_array(numpy_image: List[int]):
    try:
        numpy_image = np.array(numpy_image)

        return {"object_detector_result": detect_objects(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/detect_objects_from_base64")
async def detect_from_base64(request: DetectionRequest):
    try:
        decoded_image = base64.b64decode(request.image_path.split(',')[1])
        image = Image.open(io.BytesIO(decoded_image)).convert('RGB')
        numpy_image = np.array(image)

        return {"object_detector_result": detect_objects(numpy_image)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")