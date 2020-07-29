from picamera import PiCamera
from time import sleep
 
camera = PiCamera()

camera.start_preview(fullscreen=False, window = (100, 20, 640, 480))
sleep(20)
camera.stop_preview()
