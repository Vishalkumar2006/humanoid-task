import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame, conf_threshold=0.2, nms_threshold=0.4):
    """
    Detect objects in the frame using YOLO.
    Returns bounding boxes for detected objects with confidence > conf_threshold.
    """
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
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    bboxes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            bboxes.append((classes[class_ids[i]], (x, y, x + w, y + h), confidences[i]))
    return bboxes

def is_continuous_contact(video_path, contact_threshold=5, start_fraction=0.8):
    """
    Determine if there is continuous contact between hand and bottle in a video.
    Displays the video with bounding boxes and the result.

    Parameters:
    - video_path: Path to the video file.
    - contact_threshold: Number of frames with continuous contact to consider it manual.
    - start_fraction: The fraction of the video length after which to start checking for contact.

    Returns:
    - "natural" if the flip is natural, "manual" otherwise.
    """

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * start_fraction)
    continuous_contact_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        bboxes = detect_objects(frame)
        hand_bbox = None
        bottle_bbox = None

        for label, bbox, confidence in bboxes:
            if label == "person":
                hand_bbox = bbox
            elif label == "bottle":
                bottle_bbox = bbox

            # Draw bounding boxes on the frame
            (startX, startY, endX, endY) = bbox
            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if hand_bbox and bottle_bbox:
            hx, hy, hx2, hy2 = hand_bbox
            bx, by, bx2, by2 = bottle_bbox

            if (hx < bx2 and hx2 > bx and
                hy < by2 and hy2 > by):
                continuous_contact_frames += 1
            else:
                continuous_contact_frames = 0
        else:
            continuous_contact_frames = 0

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        if continuous_contact_frames >= contact_threshold:
            cap.release()
            cv2.destroyAllWindows()
            return "manual"

    cap.release()
    cv2.destroyAllWindows()
    return "natural"

# Example usage
video_path = "bottle video.mp4"  # Path to your video file
result = is_continuous_contact(video_path, contact_threshold=5, start_fraction=0.8)
print(result)  # Output: "natural" or "manual"
