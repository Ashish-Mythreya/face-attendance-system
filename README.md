# Face Recognition Attendance System

A real-time face recognition-based attendance system built with Python. It uses a webcam to detect and recognize faces from a predefined image dataset (`imgs/`), logs attendance in `Attendance.csv` with a 5-minute cooldown, and displays Tkinter popups and an attendance table.

## Features
- Real-time face recognition using DeepFace (Facenet model).
- Logs attendance with name and timestamp.
- 5-minute cooldown to prevent duplicate entries.
- Tkinter GUI for attendance confirmation and table display.
- Press 'g' to view attendance table, 'q' to exit.

## Requirements
- Python 3.9
- Libraries: `tensorflow-cpu==2.10.0`, `numpy==1.26.4`, `pandas==2.0.3`, `deepface`, `opencv-python`
- Images in `imgs/` (JPEG/PNG, named `FirstName_LastName.jpg`)

## Installation
```bash
pip install tensorflow-cpu==2.10.0 numpy==1.26.4 pandas==2.0.3 deepface opencv-python