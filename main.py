import ultralytics
import torch
from ultralytics import YOLO
import time
import cv2
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Check if CUDA is available and use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLO model only once
model = YOLO("yolov8n.pt").to(device)  # Ensure model is moved to the correct device

# Initialize DeepSort
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

# Video input and output setup
video_path = 'Stn_HD_1_time_2024-05-14T07_30_02_000.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

class_names = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
unique_track_ids = set()
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess frame
    frame_tensor = torch.from_numpy(og_frame).to(device).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    # Apply model
    results = model(frame_tensor)

    # Check if results are valid
    if results is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class indices

        # Convert YOLO boxes to xywh format
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2  # Center x
        xywh[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2  # Center y
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # Width
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # Height

        tracks = tracker.update(xywh, scores, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            color_id = track_id % 3
            color = (0, 0, 255) if color_id == 0 else (255, 0, 0) if color_id == 1 else (0, 255, 0)

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{class_names[int(classes[0])]}-{track_id}", (int(x1) + 10, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)

        # Update vehicle count based on unique track IDs
        vehicle_count = len(unique_track_ids)

        # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        # Draw vehicle count on frame
        cv2.putText(og_frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        # Write the frame to the output video file
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        # Show the frame
        cv2.imshow("Video", cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
