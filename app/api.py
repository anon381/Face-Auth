from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
import os
import sys
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.predict import load_model
from src.preprocessing import detect_and_crop_face, preprocess_face
from src.feature_engineering import image_to_vector
from src.attendance import log_attendance, get_attendance, force_check_out

app = FastAPI(title="FaceAuth API for Next.js/Vite")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model.pkl")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.pkl")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "train.py")
DISTANCE_THRESHOLD = 0.08  # Strict threshold for HOG + Cosine distance

def verify_face_only(img_bytes):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        raise HTTPException(status_code=500, detail="Face model not found! Please register a face first.")
    
    model = load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
        
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    cropped = detect_and_crop_face(img_array)
    if cropped is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
        
    processed = preprocess_face(cropped)
    if processed is None:
        raise HTTPException(status_code=400, detail="Error processing detected face.")
        
    features = image_to_vector(processed).reshape(1, -1)
    distances, indices = model.kneighbors(features)
    mean_dist = np.mean(distances[0])
    pred = model.predict(features)[0]
    matched_name = label_map.get(pred, "Unknown")
    
    if mean_dist <= DISTANCE_THRESHOLD:
        return True, mean_dist, matched_name
    return False, mean_dist, "Unknown"

def _train_model_background():
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR
    import logging
    try:
        res = subprocess.run([sys.executable, TRAIN_SCRIPT], env=env, cwd=BASE_DIR, capture_output=True, text=True)
        if res.returncode != 0:
             print("Background training failed:\nSTDOUT:\n", res.stdout, "\nSTDERR:\n", res.stderr)
        else:
             print("Background training succeeded.")
    except Exception as e:
        print(f"Subprocess failed: {e}")

@app.post("/api/register")
async def register_face(background_tasks: BackgroundTasks, employee_id: str = Form(...), image: UploadFile = File(...)):
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    from src.preprocessing import face_cascade
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return JSONResponse(status_code=400, content={"success": False, "message": "No face detected! Make sure you are visible."})
        
    output_dir = os.path.join(BASE_DIR, "data", "raw", employee_id)
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "0.jpg"), img_array)
    for i in range(1, 100):
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-40, 40)
        aug_img = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), aug_img)
        
    flipped_img = cv2.flip(img_array, 1)
    cv2.imwrite(os.path.join(output_dir, "100.jpg"), flipped_img)
    for i in range(101, 200):
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-40, 40)
        aug_img = cv2.convertScaleAbs(flipped_img, alpha=alpha, beta=beta)
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), aug_img)

    background_tasks.add_task(_train_model_background)
    return {"success": True, "message": f"Successfully captured face for {employee_id}. Model training started in background."}

@app.post("/api/verify-face")
async def verify_face(image: UploadFile = File(...)):
    img_bytes = await image.read()
    try:
        success, dist, name = verify_face_only(img_bytes)
        if not success:
            return JSONResponse(status_code=401, content={"success": False, "message": "Face not recognized. Access Denied."})
            
        # Log attendance automatically on successful face verify
        time_logged = log_attendance(name)
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "success": True, 
            "employeeId": name,
            "name": name,
            "loginTime": f"{date_str} {time_logged}",
            "message": f"Verified successfully"
        }
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "message": str(e.detail)})

@app.post("/api/check-in")
async def check_in(image: UploadFile = File(...)):
    img_bytes = await image.read()
    try:
        success, dist, name = verify_face_only(img_bytes)
        if success:
            time_logged = log_attendance(name)
            return {"success": True, "time": time_logged, "name": name}
        return {"success": False, "message": "Verification failed"}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "message": str(e.detail)})

@app.get("/api/attendance")
async def get_attendance_history():
    df = get_attendance()
    return df.to_dict(orient="records")

@app.get("/api/attendance/export")
async def export_attendance():
    csv_path = os.path.join(BASE_DIR, "data", "attendance.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="attendance_export.csv")
    raise HTTPException(status_code=404, detail="No data found")

from pydantic import BaseModel

class LogoutRequest(BaseModel):
    name: str

@app.post("/api/logout")
async def handle_logout(req: LogoutRequest):
    time_logged = force_check_out(req.name)
    return {"success": True, "time": time_logged}
