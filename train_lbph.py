# treino_lbph.py

import cv2
import os
import numpy as np
import pickle

FACES_DIR = "datasets"
MODEL_FILE = "modelo_lbph.yml"
LABELS_FILE = "labels.pkl"

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0

print("[INFO] A treinar LBPH...")

for pessoa in os.listdir(FACES_DIR):
    pasta = os.path.join(FACES_DIR, pessoa)
    if not os.path.isdir(pasta):
        continue

    print(f"[INFO] Processando {pessoa}")

    label_map[label_id] = pessoa

    for img_nome in os.listdir(pasta):
        path = os.path.join(pasta, img_nome)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            faces.append(img)
            labels.append(label_id)

    label_id += 1

face_recognizer.train(faces, np.array(labels))
face_recognizer.save(MODEL_FILE)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(label_map, f)

print("[INFO] Treino concluído!")
