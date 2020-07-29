#!/usr/bin/python3
from io import StringIO
import subprocess
import socket
import os
import time
from datetime import datetime
from PIL import Image
import picamera
from rpi_info import name
from camera_settings import *

# Motion detection settings:
# Threshold          - how much a pixel has to change by to be marked as "changed"
# Sensitivity        - how many changed pixels before capturing an image, needs to be higher if noisy view
# ForceCapture       - whether to force an image to be captured every forceCaptureTime seconds, values True or False
# filepath           - location of folder to save photos
# filenamePrefix     - string that prefixes the file name for easier identification of files.
# diskSpaceToReserve - Delete oldest images to avoid filling disk. How much byte to keep free on disk.
# cameraSettings     - "" = no extra settings; "-hf" = Set horizontal flip of image; "-vf" = Set vertical flip; "-hf -vf" = both horizontal and vertical flip
threshold = 10
sensitivity = sensitivity_value
forceCapture = False
forceCaptureTime = 60 * 60 # Once an hour
filenamePrefix = name
filepath = "/home/pi/APAPORIS/CURRENT/"
moved_path = "/home/pi/APAPORIS/MOVED/"
video_duration = 30
diskSpaceToReserve = 40 * 1024 * 1024 # Keep 40 mb free on disk
cameraSettings = ""

# Test-Image settings
testWidth = 200
testHeight = 150

# this is the default setting, if the whole image should be scanned for changed pixel
testAreaCount = 1
# [ [[start pixel on left side,end pixel on right side],[start pixel on top side,stop pixel on bottom side]] ]
testBorders = [ [[1,testWidth],[1,testHeight]] ]
debugMode = True

# Capture a small test image (for motion detection)
def captureTestImage(settings, width, height):
    command = "raspistill {} -w {} -h {} -t 200 -e bmp -n -o -".format(settings, width, height)
    output = subprocess.check_output(command, shell=True)
    im = Image.frombytes(mode="RGB",size=(width,height),data=output)
    buffer = im.load()
    return im, buffer

def make_video():
    global filepath
    global moved_path
    with picamera.PiCamera() as camera:
        #change these values in camera_settings on github and push to all pis for quick universal changes
        camera.rotation = camera_rotation
        camera.resolution = "720p"
        camera.brightness = camera_brightness
        camera.sharpness = camera_sharpness
        camera.contrast = camera_contrast
        camera.awb_mode = camera_awb_mode
        camera.iso = camera_ISO
        camera.color_effects = camera_color_effects
        camera.framerate = camera_framerate
        camera.exposure_mode, _shutter_speed = set_exposure_shutter(hour)
        filename = "{}_{}.h264".format(filenamePrefix,datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        camera.annotate_text_size = 15
        camera.start_recording(filepath + filename)
        start = datetime.now()
        while (datetime.now()-start).seconds < video_duration:
            camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')        	
            camera.wait_recording(0.5)
        camera.stop_recording()
        
        os.rename(filepath + filename, moved_path + filename)
        print(filename)

while True:
    n_time = datetime.now()
    hour = n_time.hour
    if hour >= puzzle_start and hour < puzzle_end:
        # Count changed pixels
        changedPixels = 0
        takePicture = False
        
        # Get first image
        image1, buffer1 = captureTestImage(cameraSettings, testWidth, testHeight)
        
        time.sleep(.1)

        # Get comparison image
        image2, buffer2 = captureTestImage(cameraSettings, testWidth, testHeight)

        if (debugMode): # in debug mode, save a bitmap-file with marked changed pixels and with visible testarea-borders
            debugimage = Image.new("RGB",(testWidth, testHeight))
            debugim = debugimage.load()

        # = xrange(0,1) with default-values = z will only have the value of 0 = only one scan-area = whole picture
        for z in range(0, testAreaCount):
            # = xrange(0,100) with default-values
            for x in range(testBorders[z][0][0]-1, testBorders[z][0][1]):
                # = xrange(0,75) with default-values; testBorders are NOT zero-based, 
                #buffer1[x,y] are zero-based (0,0 is top left of image, testWidth-1,testHeight-1 is botton right)
                for y in range(testBorders[z][1][0]-1, testBorders[z][1][1]):
                    if (debugMode):
                        debugim[x,y] = buffer2[x,y]
                        if ((x == testBorders[z][0][0]-1) or 
                        (x == testBorders[z][0][1]-1) or 
                        (y == testBorders[z][1][0]-1) or (y == testBorders[z][1][1]-1)):
                            # print "Border %s %s" % (x,y)
                            debugim[x,y] = (0, 0, 255) # in debug mode, mark all border pixel to blue
                    # Just check green channel as it's the highest quality channel
                    pixdiff = abs(buffer1[x,y][1] - buffer2[x,y][1])
                    if pixdiff > threshold:
                        changedPixels += 1
                        if (debugMode):
                            debugim[x,y] = (0, 255, 0) # in debug mode, mark all changed pixel to green
                    # Save an image if pixels changed
                    if (changedPixels > sensitivity):
                        takePicture = True # will shoot the photo later
                    if ((debugMode == False) and (changedPixels > sensitivity)):
                        break  # break the y loop
                if ((debugMode == False) and (changedPixels > sensitivity)):
                    break  # break the x loop
            if ((debugMode == False) and (changedPixels > sensitivity)):
                break  # break the z loop

        if (debugMode):
            debugimage.save(filepath + "debug.bmp") # save debug image as bmp
            #print("debug.bmp saved, {} changed pixel".format(changedPixels))

        # Check force capture
        if forceCapture:
            if time.time() - lastCapture > forceCaptureTime:
                takePicture = True

        if takePicture:
            lastCapture = time.time()
            print("Recording video @ {}".format(n_time))
            make_video()
    else:
        pass
