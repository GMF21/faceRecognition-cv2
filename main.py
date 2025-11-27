# main.py
import pygame
import sys
import time
import cv2
import numpy as np
import json
import pickle

import cfg          # Constantes e configurações
import train        # Captura de rostos
import train_lbph   # Treino LBPH

# ----------------------------
# Detectar webcam automaticamente
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
# Configuração PIN
# ----------------------------
if cfg.CONFIG_FILE.exists():
    config = json.loads(cfg.CONFIG_FILE.read_text())
else:
    config = {"pin": "1234"}

def save_config():
    cfg.CONFIG_FILE.write_text(json.dumps(config, indent=4))

# ----------------------------
# Input de texto via Pygame
# ----------------------------
def text_input_box(screen, font, prompt, mask=False):
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
        display_txt = '*' * len(txt) if mask else txt
        screen.blit(font.render(display_txt, True, cfg.COLORS['text']), (50, 260))
        pygame.display.flip()
        clock.tick(cfg.FPS)
    return txt.strip()

# ----------------------------
# Classe botão
# ----------------------------
class Button:
    def __init__(self, text, x, y, w, h, click_sound=None):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.click_sound = click_sound

    def draw(self, surf, font):
        mx, my = pygame.mouse.get_pos()
        color = cfg.COLORS['button_hover'] if self.rect.collidepoint(mx, my) else cfg.COLORS['button']
        pygame.draw.rect(surf, color, self.rect)
        surf.blit(font.render(self.text, True, cfg.COLORS['text']), (self.rect.x+10, self.rect.y+10))

    def clicked(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    if self.click_sound:
                        self.click_sound.play()
                    return True
        return False

# ----------------------------
# Carregar modelo LBPH e labels
# ----------------------------
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except:
    face_recognizer = None

if cfg.MODEL_FILE.exists() and face_recognizer:
    face_recognizer.read(str(cfg.MODEL_FILE))
else:
    print("[WARN] Modelo não encontrado ou OpenCV contrib não instalado.")
    face_recognizer = None

if cfg.LABELS_FILE.exists():
    with open(cfg.LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)
else:
    label_map = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Sons
# ----------------------------
pygame.mixer.init()
sound_click_path = cfg.ASSETS_DIR / "Wooden Button Click Sound Effect [T_Q3M6vpCAQ].wav"
sound_unlock_path = cfg.ASSETS_DIR / "microsoft-windows-xp-startup-sound.wav"
click_sound = pygame.mixer.Sound(str(sound_click_path)) if sound_click_path.exists() else None
unlock_sound = pygame.mixer.Sound(str(sound_unlock_path)) if sound_unlock_path.exists() else None

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
    pygame.display.set_caption("Face Lock LBPH")
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    # Botões
    btn_recognize = Button("Reconhecer", 650, 100, 200, 50, click_sound)
    btn_code = Button("Alterar Código", 650, 180, 200, 50, click_sound)
    btn_add_person = Button("Adicionar Pessoa", 650, 260, 200, 50, click_sound)
    btn_train = Button("Treinar LBPH", 650, 340, 200, 50, click_sound)
    btn_exit = Button("Sair", 650, 420, 200, 50, click_sound)

    cap = get_working_camera()
    if cap is None:
        pygame.quit()
        sys.exit()

    recognized_name = None
    timer_start = None
    msg_timer = None
    msg_color = cfg.COLORS['text']

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype='uint8')

        screen.fill(cfg.COLORS['bg'])
        pygame.draw.rect(screen, cfg.COLORS['panel'], pygame.Rect(600, 0, 300, cfg.HEIGHT))

        # Botões
        btn_recognize.draw(screen, font)
        btn_code.draw(screen, font)
        btn_add_person.draw(screen, font)
        btn_train.draw(screen, font)
        btn_exit.draw(screen, font)

        # BOTÃO SAIR
        if btn_exit.clicked(events):
            save_config()
            running = False

        # ALTERAR PIN
        if btn_code.clicked(events):
            timer_start = None
            recognized_name = None
            new_code = text_input_box(screen, font, "Novo Código:", mask=True)
            if new_code:
                config['pin'] = new_code
                save_config()

        # ADICIONAR PESSOA
        if btn_add_person.clicked(events):
            timer_start = None
            recognized_name = None
            success = train.main(cap, screen, font)
            if success:
                recognized_name = "Captura concluída"
                msg_color = cfg.COLORS['correct']
                msg_timer = time.time()

        # TREINAR LBPH
        if btn_train.clicked(events):
            timer_start = None
            recognized_name = None
            train_lbph.main()
            recognized_name = "Treino concluído"
            msg_color = cfg.COLORS['correct']
            msg_timer = time.time()

        # INICIAR RECONHECIMENTO
        if btn_recognize.clicked(events):
            recognized_name = None
            timer_start = time.time()

        # Mostrar câmera
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
        screen.blit(surf, (0,0))

        # RECONHECIMENTO FACIAL
        if timer_start and face_recognizer:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                rosto = gray[y:y+h, x:x+w]
                id_pred, conf = face_recognizer.predict(rosto)
                color = cfg.COLORS['correct'] if conf < 85 else cfg.COLORS['incorrect']
                name = label_map.get(id_pred, "Desconhecido") if conf < 85 else "Desconhecido"

                x_pygame = int(x * cfg.DISPLAY_WIDTH / 640)
                y_pygame = int(y * cfg.DISPLAY_HEIGHT / 480)
                w_pygame = int(w * cfg.DISPLAY_WIDTH / 640)
                h_pygame = int(h * cfg.DISPLAY_HEIGHT / 480)

                pygame.draw.rect(screen, color, (x_pygame, y_pygame, w_pygame, h_pygame), 2)
                screen.blit(font.render(name, True, color), (x_pygame, max(y_pygame-20,0)))

            # Após 10s, pedir código se ninguém reconhecido
            elapsed = time.time() - timer_start
            if elapsed >= 10 and not recognized_name:
                timer_start = None
                recognized_name = None
                pin = text_input_box(screen, font, "Código:", mask=True)
                if pin == config["pin"]:
                    recognized_name = "Código Correto"
                    msg_color = cfg.COLORS['correct']
                    if unlock_sound:
                        unlock_sound.play()
                else:
                    recognized_name = "Código Incorreto"
                    msg_color = cfg.COLORS['incorrect']
                msg_timer = time.time()

        # Mensagem
        if recognized_name:
            pygame.draw.rect(screen, (0,0,0), (600, 25, 300, 50))
            screen.blit(font.render(recognized_name, True, msg_color), (620, 30))
            if msg_timer and (time.time() - msg_timer > 5):
                recognized_name = None
                msg_timer = None

        pygame.display.flip()
        clock.tick(cfg.FPS)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
