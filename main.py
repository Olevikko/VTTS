import multiprocessing as mp
import time
import cv2
import numpy as np
import dxcam
import easyocr
import io
import pygame
from  difflib import SequenceMatcher
from global_hotkeys import *


# ---------------------------------------------------------
# 1. АЛГОРИТМ ПОРІВНЯННЯ ТЕКСТУ (ДЛЯ СКРОЛІНГУ)
# ---------------------------------------------------------

def get_new_text_only(old_text, new_text):
    """
    Визначає перекриття між кінцем старого тексту та початком нового.
    Повертає лише ту частину нового тексту, яка ще не була прочитана.
    """
    if not old_text:
        return new_text
    # Використовуємо SequenceMatcher для знаходження збігів
    matcher = SequenceMatcher(None, old_text, new_text)
    match = matcher.find_longest_match(0, len(old_text), 0, len(new_text))
    # Якщо знайдено значний збіг на початку нового тексту
    if match.size > 5 and match.b ==0:
        return new_text[match.size:].strip()

    return new_text


# ==========================================
# 2. МОДУЛЬ ГОЛОСУ (Нейромережевий TTS)
# ==========================================
def tts_worker(text_queue, command_queue):
    """
    Процес генерації та асинхронного відтворення українського мовлення.
    """
    from ukrainian_tts.tts import TTS, Voices, Stress
    print("[Голос] Завантаження нейромережі Ukrainian TTS. Зачекайте...")
    tts = TTS(device="cuda")

    # Ініціалізація аудіоплеєра для асинхронного відтворення
    pygame.mixer.init()
    print("[Голос] Модуль мовлення успішно запущено!")

    while True:
        # 1. Перевірка команд керування (високий пріоритет)
        if not command_queue.empty():
            cmd = command_queue.get()
            if cmd == "STOP":
                pygame.mixer.music.stop()
                # Очищаємо чергу тексту, щоб скасувати старі репліки
                while not text_queue.empty():
                    text_queue.get()
            elif cmd == "EXIT":
                pygame.mixer.music.stop()
                break

        # 2. Читання та озвучення нового тексту
        if not text_queue.empty():
            text_to_speak = text_queue.get()
            if text_to_speak.strip():
                print(f"[Озвучую]: {text_to_speak}")

                try:
                    audio_buffer = io.BytesIO()

                    # Генерація голосу.
                    # Voices.Mykyta.value - чоловічий, Voices.Tetiana.value - жіночий
                    tts.tts(text_to_speak, Voices.Mykyta.value, Stress.Dictionary.value, audio_buffer)
                    audio_buffer.seek(0)

                    # Відтворюємо звук не блокуючи систему
                    pygame.mixer.music.load(audio_buffer)
                    pygame.mixer.music.play()

                    # Цикл очікування завершення аудіо
                    while pygame.mixer.music.get_busy():
                        if not command_queue.empty():
                            cmd = command_queue.get()
                            if cmd == "STOP":
                                pygame.mixer.music.stop()
                                while not text_queue.empty():
                                    text_queue.get()
                                break
                            elif cmd == "EXIT":
                                pygame.mixer.music.stop()
                                return
                        time.sleep(0.05)
                except Exception as e:
                    print(f"[Помилка голосу]: {e}")

        time.sleep(0.05)

# ---------------------------------------------------------
# 3. МОДУЛЬ ЗОРУ ТА OCR (VISION PROCESS)
# ---------------------------------------------------------

def vision_worker(roi_coords, text_queue, command_queue):
    """
    Оновлений процес комп'ютерного зору з використанням прямого захоплення.
    """
    print("[Зір] Ініціалізація нейромережі EasyOCR. Зачекайте...")
    reader = easyocr.Reader(['uk', 'en'], gpu=True)
    camera = dxcam.create(output_color="BGR", backend="winrt")

    history_text = ""
    is_paused = False

    x, y, w, h = roi_coords
    print(f"[Зір] Моніторинг зони запущено: {roi_coords}")

    while True:
        if not command_queue.empty():
            cmd = command_queue.get()
            if cmd == "PAUSE_VISION":
                is_paused = not is_paused
                print(f"[Зір] Читання екрану {'ПРИЗУПИНЕНО' if is_paused else 'ВІДНОВЛЕНО'}")
            elif cmd == "EXIT":
                break

        if not is_paused:
            frame = camera.grab()
            if frame is not None:
                roi_frame = frame[int(y):int(y + h), int(x):int(x + w)]

                # Фільтрація для ідеального чорно-білого контрасту
                roi_frame = cv2.resize(roi_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Читаємо текст цілим блоком (абзацом)
                results = reader.readtext(binary, detail=0, paragraph=True)
                new_text = " ".join(results).strip()

                if new_text and len(new_text) > 3:
                    # ГОЛОВНИЙ ФІКС: Оцінюємо відсоток схожості між старим і новим текстом
                    similarity = SequenceMatcher(None, history_text, new_text).ratio()

                    # Якщо текст змінився більше ніж на 15% (це справжня зміна, а не шум)
                    if similarity < 0.85:
                        unique_text = new_text

                        # Перевіряємо на скролінг: чи є перекриття зі старим текстом?
                        if history_text:
                            matcher = SequenceMatcher(None, history_text, new_text)
                            match = matcher.find_longest_match(0, len(history_text), 0, len(new_text))

                            # Якщо знайшли спільний текст і він знаходиться на початку нового кадру
                            if match.size > 10 and match.b < 15:
                                # Відрізаємо старе, залишаємо тільки нове
                                unique_text = new_text[match.b + match.size:].strip()

                        # Відправляємо на озвучку лише якщо є що читати
                        if len(unique_text) > 3:
                            text_queue.put(unique_text)
                            history_text = new_text

        # Пауза для зняття навантаження з відеокарти
        time.sleep(0.3)

# ---------------------------------------------------------
# 4. ГОЛОВНИЙ ОРКЕСТРАТОР (MAIN PROCESS)
# ---------------------------------------------------------

def main():
    # КРОК 1: Інтерактивний вибір зони (ROI)
    print(" Роблю знімок екрану для налаштування...")
    temp_camera =dxcam.create(output_color="BGR")
    frame = temp_camera.grab()
    del temp_camera

    print("Будь ласка, виділіть зону з текстом (Діалоги) і натисніть SPACE або ENTER.")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0,0,0,0):
        print("Зону не вибрано, вихід...")
        return

    # Крок 2. Налаштування багатопроцесорності
    text_queue = mp.Queue()
    vision_cmd_queue = mp.Queue()
    tts_cmd_queue = mp.Queue()

    vision_process = mp.Process(target=vision_worker, args=(roi, text_queue, vision_cmd_queue))
    tts_process = mp.Process(target=tts_worker, args=(text_queue, tts_cmd_queue))

    vision_process.start()
    tts_process.start()

    # КРОК 3: Налаштування глобальних гарячих клавіш
    is_alive = True

    def mute_audio():
        print("[Головний процес] Натиснуто MUTE зупиняю поточний текст")
        tts_cmd_queue.put("STOP")

    def toggle_vision():
        vision_cmd_queue.put("PAUSE_VISION")

    def exit_app():
        nonlocal is_alive
        print("[Головний процес] Завершення роботи...")
        tts_cmd_queue.put("EXIT")
        vision_cmd_queue.put("EXIT")
        stop_checking_hotkeys()
        is_alive = False

    bindings = [["control", "numpad_0"], None, mute_audio], [["control", "numpad_9"], None, toggle_vision], [["control", "numpad_8"], None, exit_app]


    register_hotkeys(bindings)
    start_checking_hotkeys()

    print("\n--- СИСТЕМА ПРАЦЮЄ ---")
    print("Ctrl+Num0: Зупинити голос")
    print("Ctrl+Num9: Пауза зору")
    print("Ctrl+Num8: Вихід з програми")

    # Головний цикл підтримує роботу скрипта
    while is_alive:
        time.sleep(0.1)

    vision_process.join()
    tts_process.join()

if __name__ == '__main__':
    # Обов'язково для Windows при використанні multiprocessing
    mp.freeze_support()
    main()
