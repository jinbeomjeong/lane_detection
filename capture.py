import imagezmq, socket, cv2
import numpy as np
from mss import mss


sender = imagezmq.ImageSender(connect_to='tcp://192.168.137.1:5555')

sct = mss()
monitor_number = 2
mon = sct.monitors[monitor_number]
monitor = {'top': mon['top'], 'left': mon['left'], 'width': 960, 'height': 540, 'mon': monitor_number}

i = 0
while True:
	frame = np.array(sct.grab(monitor))
	retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
	sender.send_jpg(socket.gethostname(), frame)
	i += 1
