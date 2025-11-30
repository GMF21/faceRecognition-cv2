# train.py
import pygame
import cv2
import sys
import os
from pathlib import Path
import cfg

# ----------------------------
# Input de texto usando Pygame
# ----------------------------
def text_input_box(screen, font, prompt):
    txt = ""
    entering = True
    clock = pygame.time.Clock()
    while entering:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    entering = False
                elif event.key == pygame.K_BACKSPACE:
                    txt = txt[:-1]
                else:
                    txt += event.unicode

        screen.fill(cfg.COLORS['bg'])
        screen.blit(font.render(prompt, True, cfg.COLORS['text']), (50, 200))
        screen.blit(font.render(txt, True, cfg.COLORS['text']), (50, 260))
        pygame.display.flip()
        clock.tick(cfg.FPS)
    return txt.strip()

# ----------------------------
# Captura de caras
# ----------------------------
def capture_faces(name, cap, screen, font):
    person_dir = cfg.DATASETS_DIR / name
    person_dir.mkdir(exist_ok=True)

    existing = len(os.listdir(person_dir))
    count = existing

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    clock = pygame.time.Clock()

    while count - existing < cfg.NUM_IMAGES_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            count += 1
            cv2.imwrite(str(person_dir / f"{name}_{count}.jpg"), rosto)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{cfg.NUM_IMAGES_PER_PERSON}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Mostrar câmera no Pygame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
        screen.blit(surf, (0,0))
        pygame.display.flip()
        clock.tick(cfg.FPS)

    print("[INFO] Captura concluída!")

# ----------------------------
# Main do train
# ----------------------------
def main(cap, screen, font):
    name = text_input_box(screen, font, "Nome da pessoa:")
    if name:
        capture_faces(name, cap, screen, font)
        return True
    return False
