from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

import numpy as np

import time
import cv2
import os

LED_GRN = 27
LED_RED = 17
servo1_pin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo1_pin, GPIO.OUT)
GPIO.setup(LED_GRN, GPIO.OUT)
GPIO.setup(LED_RED, GPIO.OUT)
servo = GPIO.PWM(servo1_pin, 50)
servo.start(2)

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.85:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

	
def get_temperature():
	temp = sensor.get_object_1()
	temp_ok = temp <= 37.5
	return temp_ok, temp

def handle_door(open):
	if(open):
		print('[INFO] Opening door')
		GPIO.output(LED_GRN, GPIO.HIGH)
		GPIO.output(LED_RED, GPIO.LOW)
		servo.ChangeDutyCycle(12)
		time.sleep(3)
	else:
		print('[INFO] Closing door')
		GPIO.output(LED_RED, GPIO.HIGH)
		GPIO.output(LED_GRN, GPIO.LOW)
		servo.ChangeDutyCycle(2)
	return True


def main():
	prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
	weightsPath = os.path.sep.join(['face_detector',
		"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model("mask_detector.model")

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	camera = cv2.VideoCapture(0)
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		door = False
		face = False
		mask_ok, temp_ok = False, False
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		ret, frame = camera.read()
		if(not ret):
			print("failed to grab frame")
			break

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		# print(locs, preds)
		print('[INFO] Face detected!')
		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			mask_ok = mask > withoutMask
			temp_ok, temp = get_temperature()
			face = True

			label = "Mask" if mask_ok else "No Mask"
			color = (0, 255, 0) if mask_ok and temp_ok else (0, 0, 255)
				
			# include the probability in the label
			label = "T: {0:.2f}".format(temp)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			if(mask_ok and temp_ok):
				cv2.putText(frame, 'You are good to go!', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2 )
			elif(not temp_ok):
				cv2.putText(frame, 'Temperature not ok!', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2 )
			elif(not mask_ok):
				cv2.putText(frame, 'No mask!', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2 )
		if(not face):
			cv2.putText(frame, 'Face to the camera, please', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 255, 0), 2 )
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		door = handle_door(mask_ok and temp_ok)
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		if(not door):
			print('[INFO] Closing door')
			GPIO.output(LED_RED, GPIO.HIGH)
			GPIO.output(LED_GRN, GPIO.LOW)
			servo.ChangeDutyCycle(2)
			
	# do a bit of cleanup
	cv2.destroyAllWindows()
	bus.close()
	servo.stop()
	GPIO.cleanup()

if(__name__ == '__main__'):
	main()