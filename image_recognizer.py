from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder, Quality
import time


picam2 = Picamera2()



preview_config = picam2.create_preview_configuration(transform=Transform(hflip=True,vflip=True))

def capture_vid():
	for i in range(20):
		picam2.start_and_record_video(("DATA_SET/val/nothing/nothing" + str(i) + ".mp4"), duration=3,config=preview_config,quality=Quality.HIGH)

time.sleep(10)
capture_vid()
