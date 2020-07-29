#!/usr/bin/python3

from datetime import datetime

camera_rotation = 0
camera_resolution = "1080p"
camera_color_effects = (128,128)
focus_zoom = (0.25, 0.25, 0.5, 0.5)
feeder_zoom = (0, 0, 1, 1)
observ_zoom = (0, 0, 1, 1)
social_zoom = (0, 0, 1, 1)
resize_scale = .7
camera_ISO = 0
camera_brightness = 40
camera_shutter_speed = 3500
camera_framerate = 30
camera_exposure_mode = 'auto'
camera_awb_mode = 'tungsten'
camera_sharpness = 30
camera_contrast = 25
feeder_start = 7
feeder_end = 18
social_start = 11
social_end = 16
puzzle_start = 7
puzzle_end = 18
observ_start = 7
observ_end = 18
sensitivity_value = 200

def set_exposure_shutter(hour):
    if hour < 8:
        camera_exposure_mode = 'auto'
        camera_shutter_speed = 4500    
    elif hour < 10 and hour >=8:
        camera_exposure_mode = 'auto'
        camera_shutter_speed = 3500
    elif hour >= 10 and hour < 15:
        camera_exposure_mode = 'auto'
        camera_shutter_speed = 2500
    elif hour >= 15 and hour < 19:
        camera_exposure_mode = 'auto'
        camera_shutter_speed = 3500

    return camera_exposure_mode, camera_shutter_speed
