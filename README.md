Mediapipe + Pygame face recognition app
Single-file app that provides:
- 900x600 pygame window
- Left: camera view (600x600)
- Right: 300x600 control panel with 4 buttons: Unlock, Change Code, Add Person, Exit
- "Add Person" runs a training capture: asks for name (pygame text input), opens camera and saves 250 face images to datasets/<name>/
- Simple recognition: uses mediapipe FaceDetection to detect face, crops, resizes (100x100 grayscale) and compares with stored dataset using a simple KNN (Euclidean) on flattened images.
- If no face is recognized within 10 seconds while trying to Unlock, prompts for code (PIN). Also prompts for code if no camera is detected.
- Stores PIN and simple settings in config.json


Dependencies: mediapipe, pygame, opencv-python, numpy


Notes:
- This is a minimal, educational implementation. For production/robust recognition use proper face embeddings (dlib, face_recognition, deep models).
- If you ran into pip dependency errors (numpy/protobuf), create a fresh virtualenv and install:
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install numpy==2.2.3 opencv-python mediapipe pygame
(adjust versions to your platform)


Run: python mediapipe_pygame_face_app.py
