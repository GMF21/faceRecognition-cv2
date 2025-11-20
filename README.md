Face Recognition App (OpenCV + Pygame + LBPH)

A Python application for face recognition using OpenCV (LBPH) with a Pygame GUI.

Features

900×600 px Pygame window

Left (600×600): webcam feed

Right (300×600): control panel with buttons:

Unlock: tries to recognize the face

Change Code: modify stored PIN

Add Person: captures face images for training new users

Exit: closes the app

Add Person

Prompts for name via Pygame text input

Captures webcam images (crops detected faces)

Saves 50 face images to datasets/<name>/ for training

LBPH Training

Separate script: treino_lbph.py

Loops through all directories in datasets/ to train LBPH recognizer

Saves trained model as modelo_lbph.yml and labels as labels.pkl

Face Recognition

Detects faces using Haar Cascade

Predicts identity with LBPH recognizer

Low-confidence predictions are labeled as “Unknown”

Fallback PIN

If no face is recognized within 10 seconds after clicking “Unlock”, prompts for a PIN

PIN is stored in config.json and can be changed in the app

Correct PIN unlocks the app (“Welcome PIN OK”), otherwise shows an error

Sounds

Button click sound

Unlock sound when face or PIN is recognized

Sounds are loaded from assets/ using relative paths

Folder Structure
