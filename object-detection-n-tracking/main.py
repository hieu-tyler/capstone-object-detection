import jetson.inference
import jetson.utils

import argparse
import sys
import time
import imutils
from collections import deque

import cv2
import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="/home/nano/Videos/MVI_1470_VIS.avi",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="/home/nano/Videos/output/output.mp4",
                    nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="minimum detection threshold to use")
parser.add_argument("-b", "--buffer", type=int,
                    default=32, help="max buffer size")
parser.add_argument("--output_mode", type = str, default="normal", help="output stream type. e.g. 'normal', 'stack', 'debug'")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# ------------------------------------------------------------------------------------------
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=32)
counter = 0
(dX, dY) = (0, 0)
direction = ""
area = 0
frame_number = 0
# allow the cemra or video file to warm up
time.sleep(2.0)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# # create a copy of img to opencv img
	array_img = jetson.utils.cudaToNumpy(img)
	array_img = cv2.cvtColor(array_img, cv2.COLOR_RGB2BGR)
	array_copy = array_img.copy()

	gray_array_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)
	gray_mask_img = cv2.erode(gray_array_img, None, iterations=2)
	gray_mask_img = cv2.dilate(gray_array_img, None, iterations=2)
	
	gray_stack_img = np.vstack((gray_array_img, gray_mask_img))
	gray_stack_img = cv2.cvtColor(gray_stack_img, cv2.COLOR_GRAY2BGR)

	# print the detections
	# print("detected {:d} objects in image".format(len(detections)))

	data_point = []

	print("Frame number", frame_number)
	frame_number += 1


	for detection in detections:
		print(
			f"""
			[Width] {detection.Width}
			[Height] {detection.Height}
			[Center] {detection.Center}
			[Left] {detection.Left}
			[Right] {detection.Right}
			[Top] {detection.Top}
			[Area] {detection.Area}
			"""
		)
		center = tuple(int(x) for x in detection.Center)
		left = (int(detection.Left), int(detection.Top))
		right = (int(detection.Right), int(detection.Bottom))

		# cv2.circle(array_img, center, 10, (255, 0, 0), thickness=-1)
		
		cv2.rectangle(array_copy, left, right, color = (255, 0, 0), thickness= -1)

	# create a mask that give draw center on the point
	mask = cv2.inRange(array_copy, (254, 0, 0), (255, 0, 0))
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) == 1:
		if detections[0].Area < area:
			print('Smaller than the last contour')
			continue
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(array_img, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
			cv2.circle(array_img, center, 5, (0, 0, 255), -1)
	
		area = detections[0].Area
		
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")

			# ensure there is significant movement in the
			# x-direction
			if np.abs(dX) > 20:
				dirX = "East" if np.sign(dX) == 1 else "West"

			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 20:
				dirY = "North" if np.sign(dY) == 1 else "South"

			# handle when both directions are non-empty
			if dirX != "" and dirY != "":
				direction = "{}-{}".format(dirY, dirX)

			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
		cv2.line(array_img, pts[i - 1], pts[i], (0, 0, 255), thickness)
	
	counter += 1

	# show the movement deltas and the direction of movement on
    # the frame
	cv2.putText(array_img, direction, (800, 500),
				cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

	cv2.putText(array_img, "dx: {}, dy: {}".format(
		dX, dY), (10, array_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

	# stack image
	gray_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	stack_img = np.vstack((array_img, gray_img))
	stack_img  = cv2.cvtColor(stack_img, cv2.COLOR_BGR2RGB)

	stack_img = np.hstack((stack_img, gray_stack_img))

	# options 2: debugging with mask
	mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

	# revert the opencv image to cuda image
	array_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2RGB)
	
	output_img = jetson.utils.cudaFromNumpy(stack_img) 
	
	if opt.output_mode == "normal":
		output_img = jetson.utils.cudaFromNumpy(array_img)
	elif opt.output_mode == "stack":
		output_img = jetson.utils.cudaFromNumpy(stack_img)


	# render the image
	output.Render(output_img)

	# update the title bar
	# output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# # print out performance info
	# net.PrintProfilerTimes()

	if not input.IsStreaming() or not output.IsStreaming():
		break
