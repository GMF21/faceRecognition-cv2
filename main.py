# main.py
# Interface principal: desbloquear, alterar código, sair
import pygame
import sys
from pathlib import Path
import time
from deepface import DeepFace
import cv2
import numpy as np
import json

# --- Cores ---
COLORS = {
    "bg": (30, 30, 30),
    "panel": (45, 45, 45),
    "button": (70, 70, 70),
    "button_hover": (90, 90, 90),
    "text": (230, 230, 230),
    "warning": (200, 80, 80),
}

CONFIG_FILE = Path("config.json")
if CONFIG_FILE.exists():
    config = json.loads(CONFIG_FILE.read_text())
else:
    config = {"pin": "1234"}

def save_config():
    CONFIG_FILE.write_text(json.dumps(config, indent=4))

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
    return txt

class Button:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
    def draw(self, surf, font):
        mx, my = pygame.mouse.get_pos()
        if self.rect.collidepoint(mx, my):
            pygame.draw.rect(surf, COLORS['button_hover'], self.rect)
        else:
            pygame.draw.rect(surf, COLORS['button'], self.rect)
        surf.blit(font.render(self.text, True, COLORS['text']), (self.rect.x+10, self.rect.y+10))
    def clicked(self):
        if pygame.mouse.get_pressed()[0]:
            return self.rect.collidepoint(pygame.mouse.get_pos())
        return False

DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

def main():
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Face Lock App")
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    btn_unlock = Button("Desbloquear", 650, 100, 200, 50)
    btn_code = Button("Alterar Código", 650, 180, 200, 50)
    btn_exit = Button("Sair", 650, 260, 200, 50)

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

        btn_unlock.draw(screen, font)
        btn_code.draw(screen, font)
        btn_exit.draw(screen, font)

        if btn_exit.clicked():
            save_config()
            pygame.quit()
            sys.exit()

        if btn_code.clicked():
            new_code = text_input_box(screen, font, "Novo Código:")
            if new_code.strip():
                config['pin'] = new_code
                save_config()

        if btn_unlock.clicked():
            recognised = None
            start_time = time.time()
            while time.time() - start_time < 10:
                ret, frame = cap.read()
                if not ret:
                    continue
                for imgfile in DATASETS_DIR.glob("**/*.jpg"):
                    try:
                        result = DeepFace.verify(frame, str(imgfile), enforce_detection=False)
                        if result['verified']:
                            recognised = imgfile.parent.name
                            break
                    except:
                        pass
                if recognised:
                    break

            if not recognised:
                pin = text_input_box(screen, font, "Código:")
                if pin == config['pin']:
                    recognised = "PIN OK"

            if recognised:
                screen.blit(font.render(f"Bem-vindo {recognised}", True, COLORS['text']), (200, 400))
                pygame.display.flip()
                time.sleep(2)

        # Mostrar câmera
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (600,600))
        screen.blit(surf, (0,0))

        pygame.display.flip()
        clock.tick(24)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
