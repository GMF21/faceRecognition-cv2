project/
│
├─ assets/                     # Audio assets
│  ├─ Wooden Button Click Sound Effect [T_Q3M6vpCAQ].wav
│  └─ microsoft-windows-xp-startup-sound.wav
│
├─ datasets/                   # Captured face images per user
│  └─ <username>/
│      ├─ image_1.jpg
│      ├─ image_2.jpg
│      └─ ...
│
├─ config.json                 # Stored PIN and settings
├─ main.py                     # Main app for face unlock
├─ train.py                     # Captures faces for new users
└─ treino_lbph.py              # Trains LBPH face recognizer

Face Recognition Unlock System (CV2 + Pygame + LBPH)
Description

-This project is a face recognition-based unlock system using OpenCV, Pygame, and LBPH. It allows you to train new users, recognize faces in real-time, and fallback to a PIN code if recognition fails.

-The app is designed for educational purposes and provides a minimal, functional GUI for face unlock.

Features

-Window layout: 900x600 Pygame window

-Left panel (600x600): Live webcam feed

-Right panel (300x600): Control panel with buttons:

-Unlock: Attempts to recognize the face

-Change Code: Modify stored PIN

-Add Person: Capture new face images for training

-Exit: Close the application

Add Person

-Prompts for a name via Pygame text input

-Captures 50 face images from webcam

-Saves images in datasets/<name>/

LBPH Training (treino_lbph.py)

-Loops through datasets/ directories to train the LBPH recognizer

-Saves the trained model as modelo_lbph.yml

-Saves labels mapping as labels.pkl

Face Recognition

-Uses Haar Cascade to detect faces

-Recognizes identity using the LBPH model

-If confidence is low, labeled as Unknown

Fallback PIN

-If no face is recognized within 10 seconds during Unlock, prompts for PIN

-PIN is stored in config.json

-Correct PIN displays welcome message and unlocks app

-Incorrect PIN shows an error

Dependencies

-pygame
-numpy
-opencv-contrib-python
