def extract_frames(video_path, frame_rate=1):
    import cv2
    import os

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    video_capture = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1

    video_capture.release()
    return frames

def analyze_expression(frames, model):
    import numpy as np

    expressions = []
    for frame in frames:
        # Preprocess the frame for the model
        processed_frame = preprocess_frame(frame)
        expression = model.predict(processed_frame)
        expressions.append(expression)

    return np.array(expressions)

def load_video(video_path):
    import cv2

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video file {video_path}.")
    
    return video_capture