import pygame
import sys
from pathlib import Path
import time
import cv2
import numpy as np
import json
import pickle
import os

# ----------------------------
# Função para detetar webcam automaticamente
# ----------------------------
def get_working_camera(max_test=5):
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] Webcam encontrada no índice {i}")
                return cap
        cap.release()
    print("[ERRO] Nenhuma webcam funcional encontrada.")
    return None

# ----------------------------
# Configuração
# ----------------------------
CONFIG_FILE = Path("config.json")
ASSETS_DIR = Path("assets")
DATASETS_DIR = Path("datasets")
MODEL_FILE = "modelo_lbph.yml"
LABELS_FILE = "labels.pkl"

if CONFIG_FILE.exists():
    config = json.loads(CONFIG_FILE.read_text())
else:
    config = {"pin": "1234"}

# ----------------------------
# Cores
# ----------------------------
COLORS = {
    "bg": (30, 30, 30),
    "panel": (45, 45, 45),
    "button": (70, 70, 70),
    "button_hover": (90, 90, 90),
    "text": (230, 230, 230),
}

# ----------------------------
# Funções auxiliares
# ----------------------------
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
    return txt.strip()

class Button:
    def __init__(self, text, x, y, w, h, click_sound=None):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.click_sound = click_sound

    def draw(self, surf, font):
        mx, my = pygame.mouse.get_pos()
        color = COLORS['button_hover'] if self.rect.collidepoint(mx, my) else COLORS['button']
        pygame.draw.rect(surf, color, self.rect)
        surf.blit(font.render(self.text, True, COLORS['text']), (self.rect.x+10, self.rect.y+10))

    def clicked(self):
        if pygame.mouse.get_pressed()[0]:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                if self.click_sound:
                    self.click_sound.play()
                return True
        return False

# ----------------------------
# Carregar modelo LBPH e labels
# ----------------------------
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
if Path(MODEL_FILE).exists():
    face_recognizer.read(MODEL_FILE)
else:
    print("[WARN] Modelo não encontrado. Treine primeiro com treino_lbph.py")
    face_recognizer = None

if Path(LABELS_FILE).exists():
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)
else:
    label_map = {}

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Sons
# ----------------------------
pygame.mixer.init()
sound_click_path = ASSETS_DIR / "Wooden Button Click Sound Effect [T_Q3M6vpCAQ].wav"
sound_unlock_path = ASSETS_DIR / "microsoft-windows-xp-startup-sound.wav"
click_sound = pygame.mixer.Sound(str(sound_click_path)) if sound_click_path.exists() else None
unlock_sound = pygame.mixer.Sound(str(sound_unlock_path)) if sound_unlock_path.exists() else None

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Face Lock LBPH")
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    btn_recognize = Button("Reconhecer", 650, 100, 200, 50, click_sound)
    btn_code = Button("Alterar Código", 650, 180, 200, 50, click_sound)
    btn_exit = Button("Sair", 650, 260, 200, 50, click_sound)

    cap = get_working_camera()
    if cap is None:
        pygame.quit()
        sys.exit()

    recognized_name = None
    timer_start = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype='uint8')

        screen.fill(COLORS['bg'])
        pygame.draw.rect(screen, COLORS['panel'], pygame.Rect(600, 0, 300, 600))

        btn_recognize.draw(screen, font)
        btn_code.draw(screen, font)
        btn_exit.draw(screen, font)

        # BOTÃO SAIR
        if btn_exit.clicked():
            save_config()
            pygame.quit()
            sys.exit()

        # ALTERAR PIN
        if btn_code.clicked():
            new_code = text_input_box(screen, font, "Novo Código:")
            if new_code:
                config['pin'] = new_code
                save_config()

        # INICIAR RECONHECIMENTO
        if btn_recognize.clicked():
            recognized_name = None
            timer_start = time.time()

        # RECONHECIMENTO FACIAL
        if timer_start:
            elapsed = time.time() - timer_start
            if elapsed < 10 and face_recognizer:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    rosto = gray[y:y+h, x:x+w]
                    id_pred, conf = face_recognizer.predict(rosto)
                    if conf < 85:
                        recognized_name = label_map.get(id_pred, "Desconhecido")
                        if unlock_sound:
                            unlock_sound.play()
                    else:
                        recognized_name = "Desconhecido"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, recognized_name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            elif elapsed >= 10 and not recognized_name:
                pin = text_input_box(screen, font, "Código:")
                if pin == config["pin"]:
                    recognized_name = "PIN OK"
                    if unlock_sound:
                        unlock_sound.play()
                timer_start = None

        # Mensagem de boas vindas
        if recognized_name:
            screen.blit(font.render(f"Bem-vindo {recognized_name}", True, COLORS['text']), (200, 400))
            pygame.display.flip()
            time.sleep(2)
            recognized_name = None
            timer_start = None

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
