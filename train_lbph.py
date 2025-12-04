# train_lbph.py - treina modelo LBPH a partir de datasets/
import cv2
import os
import numpy as np
import pickle
from pathlib import Path
import cfg

def main():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("[ERRO] OpenCV contrib (cv2.face) não disponível:", e)
        return False

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    print("[INFO] A procurar datasets em:", cfg.DATASETS_DIR)
    for person_dir in sorted(cfg.DATASETS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        print(f"[INFO] Processando {person_name}")
        label_map[current_label] = person_name

        for img_file in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE) #transform ANpy
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    if not faces:
        print("[WARN] Nenhuma face encontrada. Treino abortado.")
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.write(str(cfg.MODEL_FILE))
    with open(cfg.LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print("[INFO] Treinamento concluído. Modelo salvo em", cfg.MODEL_FILE)
    return True

if __name__ == "__main__":
    main()
