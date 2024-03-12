import cv2
import numpy as np
import tensorflow as tf
from picamera2.picamera2 import Picamera2
import pyautogui as pg
import time
import speech_recognition as sr
from pynput.mouse import Controller
import threading
from pynput.keyboard import Key, Controller

def vol_up():
    keyboard = Controller()
    keyboard.press(Key.media_volume_up)
    keyboard.release(Key.media_volume_up)
# Load your trained model
model = tf.keras.models.load_model('/home/amsozzer/Downloads/my-model4.keras')

# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)
picam2.start()

# Mouse controller
mouse = Controller()

# Parameters
frame_count = 16
frame_height = 160
frame_width = 160
num_classes = 2

def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        processed_frames.append(frame_resized)
    processed_frames = np.array(processed_frames)
    processed_frames = np.expand_dims(processed_frames, axis=0)
    return processed_frames

def capture_frames():
    frames = []
    for _ in range(frame_count):
        frame = picam2.capture_array()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames

def model_prediction():
    while True:
        frames = capture_frames()
        frames_preprocessed = preprocess_frames(frames)
        predictions = model.predict(frames_preprocessed)
        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted class: {predicted_class}")
        if predicted_class != 0:
            pg.scroll(-50)
        time.sleep(1)  # Sleep to avoid overloading the CPU

def full_screen():
    pg.hotkey("alt", "f11")

def launch_website(website):
    pg.press('win')
    pg.moveTo(118,91,1)
    pg.click()
    pg.moveTo(275,103,1)
    pg.click()
    pg.moveTo(462,115,2)
    pg.click()
    time.sleep(2)
    pg.write(f"{website}.com")
    pg.press('enter')
    time.sleep(10)
    pg.moveTo(462,300,1)
    
def mute():
    pg.moveTo(1782,17)
    pg.click()
    pg.moveTo(1780,167)
    pg.click()
    pg.move(1782,17)
    pg.click()


def voice_commands():
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as src:
            print("Listening...")
            audio = r.listen(src)
        try:
            command = r.recognize_google(audio)
            print(f"Google thinks you said: {command}")
            if command.lower().startswith("open "):
                launch_website(command.split(" ")[1])
            elif command.lower() == "full screen":
                full_screen()
            elif command == "mute":
                mute()
            elif command == "volume up":
                vol_up()
            else:
                print("Command Not Found")
        except Exception as e:
            print("Error:", e)

def start_threads():
    model_thread = threading.Thread(target=model_prediction)
    voice_thread = threading.Thread(target=voice_commands)
    
    model_thread.start()
    voice_thread.start()
    
    model_thread.join()
    voice_thread.join()

if __name__ == "__main__":
    start_threads()


