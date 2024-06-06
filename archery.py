import cv2
import numpy as np

ball = []

# Path to your video file
video_path = "archery1 video.mp4"  # Update path to the uploaded file

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get frame width and height for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Define the HSV color range for detecting red color (adjust if needed)
lower_hue = np.array([0, 100, 100])
upper_hue = np.array([10, 255, 255])

# Read and display frames from the video
while True:
    ret, frame = cap.read()

    # Check if there are still frames to read
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    result = cv2.bitwise_or(frame, frame, mask=mask)

    # Find contours of shapes in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 15:  # Filter small contours to avoid noise
            try:
                cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)
                ball.append((int(x), int(y)))
            except Exception as e:
                print(f"Error: {e}")
                pass
            
            if len(ball) > 2:
                for i in range(1, len(ball)):
                    cv2.line(frame, ball[i-1], ball[i], (0, 0, 255), 5)
    
    out.write(frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for user input to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
