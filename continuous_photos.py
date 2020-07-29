#!/usr/bin/python3
# this program will take 600 photos continuously, with timestamp

import subprocess, socket, os, shutil, time, picamera, signal
from io import StringIO
from datetime import datetime
from PIL import Image
from rpi_info import name
from camera_settings import *
from sigterm_exception import *

filepath = "/home/pi/APAPORIS/CURRENT/"
moved_path = "/home/pi/APAPORIS/MOVED/"
filenamePrefix = name


def make_photos(hour):
    global filepath
    global moved_path
    with picamera.PiCamera() as camera:
        camera.rotation = camera_rotation
        camera.resolution = camera_resolution
        camera.zoom = feeder_zoom
        camera.brightness = camera_brightness
        camera.sharpness = camera_sharpness
        camera.contrast = camera_contrast
        camera.awb_mode = camera_awb_mode
        camera.color_effects = camera_color_effects
        camera.iso = camera_ISO
        camera.exposure_mode, camera.shutter_speed = set_exposure_shutter(hour)
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = '{}{}_{}'.format(filepath,filenamePrefix,time_stamp)
        os.mkdir(dir_name)
        camera.annotate_text_size = 15
        camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
        resize_tuple = (int(resize_scale*camera.resolution[0]),int(resize_scale*camera.resolution[1]))
        try:        
            for i, filename in enumerate(camera.capture_continuous("{}/{}_".format(dir_name,filenamePrefix)+"{timestamp:%Y-%m-%d-%H-%M-%S-%f}.jpg", resize = resize_tuple)):
                camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
                if i == 599:
                    return dir_name
        except Exception:
            print("error during photo capture")
            return dir_name

signal.signal(signal.SIGTERM, signal_handler)

try:
    while True:
        hour = datetime.now().hour
        if hour >= feeder_start and hour < feeder_end:
            dir_name = make_photos(hour)
            shutil.move(dir_name,moved_path)
        else:
            pass

except (SigTermException, KeyboardInterrupt):
    try:
        shutil.move(dir_name,moved_path)
    except:
        print("failed to move directory")
    finally:
        sys.exit()
