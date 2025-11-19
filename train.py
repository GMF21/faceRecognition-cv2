# train.py
# Treino/captura de imagens usando OpenCV + Pygame

import pygame
import sys
from pathlib import Path
import cv2
import numpy as np

# --- Cores ---
COLORS = {
    "bg": (30, 30, 30),
    "panel": (45, 45, 45),
    "button": (70, 70, 70),
    "button_hover": (90, 90, 90),
    "text": (230, 230, 230),
    "warning": (200, 80, 80),
}

# --- Funções ---
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
        screen.fill(COLORS['bg'])
        screen.blit(font.render(prompt, True, COLORS['text']), (50, 200))
        screen.blit(font.render(txt, True, COLORS['text']), (50, 260))
        pygame.display.flip()
        clock.tick(30)
    return txt.strip()

class Button:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
    def draw(self, surf, font):
        mx, my = pygame.mouse.get_pos()
        color = COLORS['button_hover'] if self.rect.collidepoint(mx, my) else COLORS['button']
        pygame.draw.rect(surf, color, self.rect)
        surf.blit(font.render(self.text, True, COLORS['text']), (self.rect.x+10, self.rect.y+10))
    def clicked(self):
        if pygame.mouse.get_pressed()[0]:
            return self.rect.collidepoint(pygame.mouse.get_pos())
        return False

# --- Diretórios ---
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

# --- Captura de imagens ---
def capture_face_images(name, cap, num_images=30):
    person_dir = DATASETS_DIR / name
    person_dir.mkdir(exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(str(person_dir / f"{count:03d}.jpg"), face_img)
            count += 1

        # Mostrar a câmera em tempo real durante a captura
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (600,600))
        screen.blit(surf, (0,0))
        pygame.display.flip()
        pygame.time.delay(30)

    print(f"[INFO] Capturadas {num_images} imagens para {name}")

# --- Main ---
def main():
    global screen  # necessário para a função de captura
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Adicionar Pessoa")
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    btn_add = Button("Adicionar Pessoa", 650, 100, 200, 50)
    btn_exit = Button("Sair", 650, 180, 200, 50)

    # abrir câmera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        sys.exit()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            frame = 255 * np.zeros((480, 640, 3), dtype='uint8')

        screen.fill(COLORS['bg'])
        pygame.draw.rect(screen, COLORS['panel'], pygame.Rect(600, 0, 300, 600))

        btn_add.draw(screen, font)
        btn_exit.draw(screen, font)

        if btn_exit.clicked():
            running = False

        if btn_add.clicked():
            name = text_input_box(screen, font, "Nome da pessoa:")
            if name:
                capture_face_images(name=name, cap=cap)  # tira 250 fotos

        # Mostrar câmera no lado esquerdo
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (600,600))
        screen.blit(surf, (0,0))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
