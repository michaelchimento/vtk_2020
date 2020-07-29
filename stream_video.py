#!/usr/bin/python3
import io
import itertools
import picamera
import logging
import socketserver
from datetime import datetime
from threading import Condition
from http import server
from rpi_info import name
from camera_settings import *
from sys import argv

#run "python3 stream_video.py 1" if you'd like to toggle on the focus zoom. Defaults to framing view.
global focus
focus = argv[1] if len(argv) > 1 else 0

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
    def service_actions(self):
        global camera
        camera.annotate_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        camera.wait_recording(0.5)

with picamera.PiCamera() as camera:
    camera.rotation = camera_rotation
    if focus:
        camera.resolution = "1080p"
        camera.zoom = focus_zoom
    else:
        camera.resolution = "720p"
    camera.brightness = camera_brightness
    camera.sharpness = camera_sharpness
    camera.contrast = camera_contrast
    camera.framerate = camera_framerate
    camera.awb_mode = camera_awb_mode
    camera.exposure_mode = camera_exposure_mode
    camera.iso = camera_ISO
    output = StreamingOutput()
    camera.annotate_text = "{}".format(name)
    camera.start_recording(output, format='mjpeg')
    try:
        if focus:        
            frame_width = camera.zoom[2]*camera.resolution[0]
            frame_height = camera.zoom[3]*camera.resolution[1]
        else:
            frame_width = camera.resolution[0]
            frame_height = camera.resolution[1]
        PAGE="<html><head><title>Greti Live Stream</title></head><body><h1>{}</h1><img src=\"stream.mjpg\" width=\"{}\" height=\"{}\" /></body></html>".format(name,frame_width, frame_height)
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    finally:
        camera.stop_recording()
