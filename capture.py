import imagezmq, socket, cv2
import numpy as np
from mss import mss

mon = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}
sct = mss()
sender = imagezmq.ImageSender('tcp://127.0.0.1:5555')

i = 0
while True:
	print(i)
	frame = np.array(sct.grab(mon))
	retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
	sender.send_jpg(socket.gethostname(), frame)
	i += 1
