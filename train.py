import pygame
import cv2
import sys
import os
from pathlib import Path
import cfg

# ----------------------------
# UI
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

class Button:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surf, font):
        mx, my = pygame.mouse.get_pos()
        color = cfg.COLORS['button_hover'] if self.rect.collidepoint(mx, my) else cfg.COLORS['button']
        pygame.draw.rect(surf, color, self.rect)
        surf.blit(font.render(self.text, True, cfg.COLORS['text']), (self.rect.x+10, self.rect.y+10))

    def clicked(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    return True
        return False

# ----------------------------
# Captura de faces
# ----------------------------
def capture_faces(name, cap, screen):
    person_dir = cfg.DATASETS_DIR / name
    person_dir.mkdir(exist_ok=True)

    existing = len(os.listdir(person_dir))
    count = existing

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print(f"[INFO] Capturando {cfg.NUM_IMAGES_PER_PERSON} imagens de {name}...")

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
def main():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
    pygame.display.set_caption("Captura de Rostos (LBPH)")
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    btn_add = Button("Adicionar Pessoa", 650, 100, 200, 50)
    btn_exit = Button("Sair", 650, 180, 200, 50)

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera.")
        pygame.quit()
        sys.exit()

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        screen.fill(cfg.COLORS['bg'])
        pygame.draw.rect(screen, cfg.COLORS['panel'], pygame.Rect(600, 0, 300, 600))

        btn_add.draw(screen, font)
        btn_exit.draw(screen, font)

        if btn_exit.clicked(events):
            running = False

        if btn_add.clicked(events):
            name = text_input_box(screen, font, "Nome da pessoa:")
            if name:
                capture_faces(name, cap, screen)

        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
            surf = pygame.transform.scale(surf, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
            screen.blit(surf, (0,0))

        pygame.display.flip()
        clock.tick(cfg.FPS)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
