# 🛡️ FaceTrack | Face Authentication & Attendance System

🚀 **Live Demo:** [https://face-auth-1-276y.onrender.com/](https://face-auth-1-276y.onrender.com/)

A modern, cloud-deployed Face Authentication and Attendance Tracking system built with Python, OpenCV, and Machine Learning. Designed for seamless browser-based employee check-ins using a webcam.

---

## ✨ Features

- **Real-Time Face Authentication:** Browser-to-server facial recognition using `getUserMedia`.
- **Biometric Security:** Uses **HOG (Histogram of Oriented Gradients)** combined with **K-Nearest Neighbors (KNN)** to securely and accurately identify users using geometric face structures rather than raw pixels.
- **Attendance Dashboard:** Automatically logs Check-ins and Check-outs upon successful facial verification.
- **Admin Management:** Secure portal to view live attendance records and manage registered employees.
- **Cloud-Ready:** Container-ready architecture deployed on Render with an ephemeral backend that automatically creates models dynamically.

## 🛠️ Tech Stack

- **Frontend:** Streamlit, HTML5/JS (for custom webcam components)
- **Backend:** FastAPI, Uvicorn
- **Machine Learning:** OpenCV (`opencv-python-headless`), Scikit-Learn, NumPy
- **Deployment:** Render.com PaaS

---

## ☁️ Cloud Deployment Note (Render Free Tier)

Because this application is deployed on Render's **Free Tier**, the server uses *Ephemeral Storage*. This means:
- If the application goes to sleep from 15 minutes of inactivity, the server spins down.
- Upon spinning back up, the hard drive is wiped clean to protect user data.
- **You will need to Register a face again if the server restarts.** 

> *In a production enterprise environment, this would be connected to an AWS S3 bucket and a persistent PostgreSQL database.*

---

## 💻 Local Development

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Face-Auth.git
cd Face-Auth
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Backend API
Start the FastAPI server (handles the core ML/Authentication pipelines):
```bash
uvicorn app.api:app --reload --port 8001
```

### 4. Run the Frontend UI
In a separate terminal, launch the Streamlit app:
```bash
streamlit run app/main.py
```
*Your application will be available locally at `http://localhost:8501/`.*

---

## 📁 Project Structure

```text
├── app/                        # Streamlit & FastAPI application layer
│   ├── main.py                 # Streamlit Frontend Entry point 
│   ├── api.py                  # FastAPI Backend Endpoints
│   └── ui.py                   # Custom UI & JavaScript Components
├── data/                       # Local Dataset storage (Gitignored)
├── models/                     # Compiled ML .pkl models (Gitignored)
├── src/                        # Core Machine Learning Pipeline
│   ├── feature_engineering.py  # HOG flattening & transformations
│   ├── preprocessing.py        # Face detection (Haar Cascades) & alignment
│   ├── train.py                # KNN Model training logic
│   ├── predict.py              # Inference & distance threshold logic
│   └── attendance.py           # Attendance logging system
├── requirements.txt            # Python Dependencies
└── config.py                   # Centralized Configuration paths
```
