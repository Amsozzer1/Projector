from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder, Quality
import time
import pyttsx3
engine = pyttsx3.init()


picam2 = Picamera2()



preview_config = picam2.create_preview_configuration(transform=Transform(hflip=True,vflip=True))
engine.say("Starting TRAINING")
engine.runAndWait()
def capture_vid():
	for i in range(60):
		picam2.start_and_record_video(("DATA_SET/train/mouse_left/" + str(i) + ".mp4"), duration=5,quality=Quality.HIGH) #config=preview_config,
		engine.say(f"{i}th Piece Recorded")
		engine.runAndWait()
		time.sleep(1)
engine.say("Starting VALIDATION")
engine.runAndWait()
def capture_vid_val():
	for i in range(20):
		picam2.start_and_record_video(("DATA_SET/val/mouse_left/" + str(i) + ".mp4"), duration=5,quality=Quality.HIGH) #config=preview_config,
		engine.say(f"{i}th Piece Recorded")
		engine.runAndWait()
		time.sleep(1)
time.sleep(4)
capture_vid()
capture_vid_val()
