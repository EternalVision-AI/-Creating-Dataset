import cv2
import os
import numpy as np
import math
from collections import Counter
from datetime import datetime
# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

recognition_classes = ['White', 'Black', 'Blue', 'Green', 'Yellow', 'Orange', 'Red', 'Purple']


confThreshold = 0.7  # Confidence threshold
nmsThreshold = 0.6 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "/ball.onnx")


# Define BGR color values for each class
class_colors = {
    'White': (255, 255, 255),
    'Black': (0, 0, 0),
    'Blue': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (0, 255, 255),
    'Orange': (0, 165, 255),
    'Red': (0, 0, 255),
    'Purple': (255, 0, 255)
}

class_points = {
    'White': [0, 0],
    'Black': [0, 0],
    'Blue': [0, 0],
    'Green': [0, 0],
    'Yellow': [0, 0],
    'Orange': [0, 0],
    'Red': [0, 0],
    'Purple': [0, 0]
}

class_points_prev = {
    'White': [0, 0],
    'Black': [0, 0],
    'Blue': [0, 0],
    'Green': [0, 0],
    'Yellow': [0, 0],
    'Orange': [0, 0],
    'Red': [0, 0],
    'Purple': [0, 0]
}
class_race = {
    'White': [-1, 0, 0],
    'Black': [-1, 0, 0],
    'Blue': [-1, 0, 0],
    'Green': [-1, 0, 0],
    'Yellow': [-1, 0, 0],
    'Orange': [-1, 0, 0],
    'Red': [-1, 0, 0],
    'Purple': [-1, 0, 0],
}
class_race_time = {
    'White': 0,
    'Black': 0,
    'Blue': 0,
    'Green': 0,
    'Yellow': 0,
    'Orange': 0,
    'Red': 0,
    'Purple': 0,
}


class_whole_time = {
    'White': 0,
    'Black': 0,
    'Blue': 0,
    'Green': 0,
    'Yellow': 0,
    'Orange': 0,
    'Red': 0,
    'Purple': 0,
}


def DetectionProcess(original_image):
	height, width, _ = original_image.shape
	length = max((height, width))
	image = np.zeros((length, length, 3), np.uint8)
	image[0:height, 0:width] = original_image
	scale = length / INPUT_WIDTH

	blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(INPUT_WIDTH, INPUT_WIDTH), swapRB=True)
	detection_model.setInput(blob)
	outputs = detection_model.forward()

	outputs = np.array([cv2.transpose(outputs[0])])
	rows = outputs.shape[1]

	boxes = []
	scores = []
	class_ids = []

	for i in range(rows):
		classes_scores = outputs[0][i][4:]
		(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
		if maxScore >= confThreshold:
			box = [
				outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
				outputs[0][i][2], outputs[0][i][3]]
			boxes.append(box)
			scores.append(maxScore)
			class_ids.append(maxClassIndex)

	result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

	detections = []
	for i in range(len(result_boxes)):
		index = result_boxes[i]
		box = boxes[index]
		detection = {
			'class_id': class_ids[index],
			'class_name': recognition_classes[class_ids[index]],
			'confidence': scores[index],
			'box': box,
			'scale': scale}
		detections.append(detection)
	return detections

def draw_class_rectangle(img, left, top, right, bottom, class_name):
    # Get the color for the class from the dictionary
    color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found
    # Draw the rectangle
    # cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    
    # Calculate the center of the rectangle
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    # Calculate the radius as half the width or height (whichever is smaller)
    radius = min((right - left) // 2, (bottom - top) // 2)
    
    # Draw the circle on the image
    cv2.circle(img, (center_x, center_y), radius, color, 2)
    return img

def draw_startline(img):
		start_point = (750, 180)
		end_point = (1120, 250)
		color = (0, 0, 255)  # Green
		thickness = 5
		cv2.line(img, start_point, end_point, color, thickness)
		return img

def rotate_point_clockwise(x, y, angle_degrees = 10.713123022791):
    # Convert the angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Calculate the new x and y coordinates using the rotation formula
    new_x = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    new_y = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    
    return new_x, new_y

isdisplayed = False
image_count_g = 5421
def DetectCard(img, image_count):
    global isdisplayed, image_count_g
    detections = DetectionProcess(img)
    detected_cards = []
    for detection in detections:
        class_id, class_name, confidence, box, scale = detection['class_id'], detection['class_name'], detection['confidence'], detection['box'], detection['scale']
        left, top, right, bottom = round(box[0] * scale), round(box[1] * scale), round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
		# Ensure bounding box is within image dimensions
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, img.shape[1] - 1)
        bottom = min(bottom, img.shape[0] - 1)
    
        detected_cards.append([left, top, right, bottom])
    if len(detected_cards) and len(detected_cards) != 8:
        image_name = f"JPEGImages/{int(image_count_g)}.jpg"
        image_count_g += 1
        cv2.imwrite(image_name, img)
        print(f"Saved {image_name}")

    cv2.imshow("Race", img)
    cv2.waitKey(1)
def process_video(video_path):
    # Open the video file
		cap = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
		if not cap.isOpened():
				print(f"Error opening video file: {video_path}")
				return

		frame_count = 0
    # Loop over each frame in the video
		while True:
				ret, frame = cap.read()
				if not ret:
						break  # Stop if no more frames are returned

				 # Get the current frame timestamp in milliseconds
				timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Convert to seconds (optional)
				timestamp_sec = timestamp_ms / 1000.0
    
				if frame_count % 5 == 0:
					DetectCard(frame, frame_count/2)
				frame_count += 1

    # Release the video capture object
		# cap.release()

if __name__ == '__main__':
    video_path = "./vid/(8).mp4"  # Change to your video path
    # Record start time
    process_video(video_path)






