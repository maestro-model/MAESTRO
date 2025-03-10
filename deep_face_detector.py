# face_detection.py




import cv2
from deepface import DeepFace
from mtcnn import MTCNN
from collections import defaultdict, Counter
import numpy as np



# Initialize MTCNN face detector
detector = MTCNN()


from collections import Counter

def process_chunk_for_face_detection(idea_frame_pairs, output_path):
    """
    Processes each chunk in idea_frame_pairs for face detection and returns aggregated detected attributes.
    """
    import cv2

    # Initialize VideoWriter object to save processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, 30, (1024, 576))  # Use the correct resolution for output

    detected_info = []  # To store aggregated info for each chunk

    for chunk_idx, pair in enumerate(idea_frame_pairs):
        segment_text = pair["text"]
        frames = [frame.astype(np.uint8) for frame in pair["frames"] if frame is not None]

        if not frames:
            print(f"Skipping chunk {chunk_idx + 1}: No frames available.")
            detected_info.append("No faces detected")
            continue

        print(f"Processing Chunk {chunk_idx + 1}: {segment_text}")

        # Counters to aggregate results
        race_counter = Counter()
        gender_counter = Counter()
        age_counter = Counter()

        for frame in frames:
            # Detect faces in the frame
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, w, h = face['box']
                try:
                    # Analyze the face
                    analysis = DeepFace.analyze(
                        frame[y:y + h, x:x + w],
                        actions=['emotion', 'race', 'gender', 'age'],
                        enforce_detection=False
                    )

                    # Update counters with detected attributes
                    race = analysis[0]['dominant_race']
                    gender_dict = analysis[0]['gender']
                    gender = max(gender_dict, key=gender_dict.get)
                    age = int(analysis[0]['age'])

                    race_counter[race] += 1
                    gender_counter[gender] += 1
                    age_counter[age] += 1

                    # Draw bounding box and label (optional)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{race}, {gender}, age {age}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error in DeepFace analysis: {e}")

            # Write processed frame to output video
            out.write(frame)

        # Aggregate results for the chunk
        most_common_race = race_counter.most_common(1)[0][0] if race_counter else "Unknown"
        most_common_gender = gender_counter.most_common(1)[0][0] if gender_counter else "Unknown"
        avg_age = sum(age * count for age, count in age_counter.items()) // sum(age_counter.values()) if age_counter else "Unknown"

        # Format aggregated result
        aggregated_result = f"{most_common_race}, {most_common_gender}, avg age {avg_age}"
        print(f"Aggregated Result for Chunk {chunk_idx + 1}: {aggregated_result}")

        # Update the chunk in idea_frame_pairs XXX DONT DO UPDATING HERE 
        #idea_frame_pairs[chunk_idx]["deepface_analysis"] = aggregated_result

        # Append to detected_info
        #detected_info.append(aggregated_result)

    # Release the VideoWriter
    out.release()
    print(f"Processed video saved at {output_path}")

    # Return the detected_info with aggregated results
    return aggregated_result


def is_same_person(face_embedding, known_embeddings, threshold=10):
    """
    Verifies if the face embedding corresponds to a known person by checking the distance threshold.
    """
    distances = [np.linalg.norm(np.array(face_embedding) - np.array(known_embedding)) for known_embedding in known_embeddings]
    return any(distance < threshold for distance in distances)

