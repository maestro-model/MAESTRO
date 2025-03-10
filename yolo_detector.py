from ultralytics import YOLO
import numpy as np

def run_yolo_on_frames(frames, model, confidence_threshold=0.5, iou_threshold=0.45):
    """
    Run YOLO object detection on a list of frames and print the detected objects and their bounding boxes.
    """
    for frame_idx, frame in enumerate(frames):
        # Convert frame to required format (e.g., BGR to RGB)
        frame_rgb = frame[:, :, ::-1]
        
        # Run YOLO detection
        results = model.predict(frame_rgb)
        
        # Print results for each frame
        print(f"Frame {frame_idx + 1}: Detected objects")
        
        for result in results:
            boxes = result.boxes  # Extract bounding boxes
            for box in boxes:
                if box.conf > confidence_threshold:
                    # Extract information from each box and print it
                    class_id = int(box.cls)  # Get the class ID (e.g., 0, 1, 2 for person, bicycle, car, etc.)
                    class_name = model.names[class_id]  # Get the class name using the class ID
                    bbox = box.xyxy.tolist()  # Get bounding box coordinates [x1, y1, x2, y2]
                    confidence = box.conf.tolist()  # Get confidence score
                    
                    # Print the detected object's details
                    print(f"  - Object: {class_name}")
                    print(f"    Bounding Box: {bbox}")
                    print(f"    Confidence: {confidence}")
                    
        print("-" * 40)  # Separator between frames for clarity


def yolo_detection_on_chunk(idea_frame_pairs, chunk_idx):
    """
    Run YOLO object detection on the specified chunk of idea_frame_pairs.
    If chunk_idx is specified, process only that chunk; otherwise, process all chunks.
    """
    model = YOLO("yolo11n.pt") 
    
    print(f"CHOSEN CHUNK FOR YOLO IS {chunk_idx+1}")
    
    if model is None:
        raise ValueError("YOLO model is not loaded.")
    
    # If chunk_idx is specified, select the corresponding chunk, otherwise process all chunks
    if chunk_idx is not None:
        print(f"Running YOLO on chunk {chunk_idx + 1}...")
        selected_chunk = [idea_frame_pairs[chunk_idx]]
    else:
        print("Running YOLO on all chunks...")
        selected_chunk = idea_frame_pairs

    # Process each chunk
    for idx, pair in enumerate(selected_chunk):
        segment_text = pair["text"]
        frames = [frame.astype(np.uint8) for frame in pair["frames"] if frame is not None]
        
        # Skip segment if no valid frames are available
        if not frames:
            print(f"Skipping segment {idx + 1}: No valid frames available.")
            pair["yolo_detections"] = "No frames available"
            continue
        
        print(f"Processing Segment: {segment_text}")
        
        # Run YOLO on the frames of the selected segment
        detections = run_yolo_on_frames(frames, model)
        
        # Store YOLO detections in the pair
        pair["yolo_detections"] = detections
    
    return idea_frame_pairs  # Return the updated idea_frame_pairs
