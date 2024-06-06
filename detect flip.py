import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Handle the output layers indices properly
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to detect bottles
def detect_bottle(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "bottle":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes

# Function to initialize tracker
def initialize_tracker(frame, box):
    tracker = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.TrackerCSRT.create()
    tracker.init(frame, tuple(box))
    return tracker

# Function to update tracker
def update_tracker(tracker, frame):
    success, box = tracker.update(frame)
    return success, box

# Function to detect flips
def detect_flip(trajectory):
    if len(trajectory) < 2:
        return False
    dy = [y2 - y1 for (x1, y1), (x2, y2) in zip(trajectory[:-1], trajectory[1:])]
    significant_dy = [d for d in dy if abs(d) > 20]  # Threshold for significant vertical movement
    if len(significant_dy) >= 2:
        return True
    return False

# Path to your video file
video_path = "bottle video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Variables to store trajectory
trajectory = []

# Read frames from the video
flip_detected = False
while True:
    ret, frame = cap.read()

    if not ret:
        # print("Failed to read frame")
        break

    try:
        boxes, indexes = detect_bottle(frame)
    except Exception as e:
        print(f"Error in detect_bottle: {e}")
        break

    # Convert indexes to a list if it's not already
    if isinstance(indexes, np.ndarray):
        indexes = indexes.flatten().tolist()

    if len(indexes) > 0:
        # Assume the first detected bottle is the one we are interested in
        box = boxes[indexes[0]]
        try:
            tracker = initialize_tracker(frame, box)
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            break

        while True:
            ret, frame = cap.read()
            if not ret:
                # print("Failed to read frame in tracker loop")
                break

            success, box = update_tracker(tracker, frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                center = (x + w // 2, y + h // 2)
                trajectory.append(center)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if detect_flip(trajectory):
                flip_detected = True
                cv2.putText(frame, "Bottle Flip Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Flip Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop if needed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
