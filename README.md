# AI Paint

An AI-based **virtual painter** that lets you draw in the air using **hand gestures** detected from your webcam.  
Built with **Python, OpenCV, and MediaPipe (Tasks API)**.

---

## Project Note
> **This project was originally created in November 2022** as a learning and experimentation project using the MediaPipe hand-tracking pipeline available at that time.  
> It has been **updated, fixed, and pushed to GitHub recently**, including compatibility with **modern Python versions and the latest MediaPipe Tasks API**.

---

## Features
- Draw in real time using your **index finger**
- Change colors using **gesture-based selection**
- Erase drawings with an **eraser gesture**
- Uses **MediaPipe Hand Landmarker (latest API)**
- Works on **macOS / Windows / Linux**

---

## Project Structure
```
AI Paint/
│── Ai_virtual_painter.py
│── handtrackingmodule.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── Header/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
│
└── models/
    └── hand_landmarker.task
```

---

## Requirements
- Python **3.12** (recommended)
- Webcam
- macOS / Windows / Linux

---

## Setup & Run

### Install Python 3.12 (macOS)
```bash
brew install python@3.12
```

Verify:
```bash
python3.12 --version
```

---

### Clone or download the project
```bash
git clone <your-repo-url>
cd "AI Paint"
```

---

### Create virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Download MediaPipe hand model
```bash
mkdir -p models
curl -L   -o models/hand_landmarker.task   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

### Run the application
```bash
python Ai_virtual_painter.py
```

---

##  How It Works
- **MediaPipe Hand Landmarker** detects 21 hand landmarks
- Finger states determine:
  - Selection mode (2 fingers up)
  - Drawing mode (index finger)
  - Eraser mode
- OpenCV renders drawings onto a live camera feed

---


Happy painting! 🎨✨
