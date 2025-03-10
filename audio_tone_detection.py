from moviepy.audio.io.AudioFileClip import AudioFileClip
import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np

# Load the pre-trained model and feature extractor for emotion recognition
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"  # Your model path
tone_model = AutoModelForAudioClassification.from_pretrained(model_name)  # Load model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")  # Use a feature extractor

# Modify the model's layers as per your requirement (e.g., adjust projector and classifier)
tone_model.projector = nn.Linear(1024, 1024, bias=True)
tone_model.classifier = nn.Linear(1024, 8, bias=True)

# If you have a custom model state dictionary, load it manually
torch_state_dict = torch.load("/home/itfelicia/project/tone_pytorch_model.bin", map_location=torch.device('cpu'))

tone_model.projector.weight.data = torch_state_dict['classifier.dense.weight']
tone_model.projector.bias.data = torch_state_dict['classifier.dense.bias']
tone_model.classifier.weight.data = torch_state_dict['classifier.output.weight']
tone_model.classifier.bias.data = torch_state_dict['classifier.output.bias']

# Emotion labels mapping
id2label = {
    "0": "angry",
    "1": "calm",
    "2": "disgust",
    "3": "fearful",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprised"
}

def analyze_tone(audio_path, start_time, end_time):
    """
    Analyze the tone (emotion) of the audio between start_time and end_time.
    """
    clip = AudioFileClip(audio_path).subclip(start_time, end_time)
    audio_array = clip.to_soundarray(fps=16000).mean(axis=1)  # Convert to mono
    audio_tensor = torch.tensor(audio_array).unsqueeze(0)  # Add batch dimension

    # Use the feature extractor to process the audio tensor
    inputs = feature_extractor(
        raw_speech=audio_tensor.numpy(),  # Convert to numpy before passing to feature extractor
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    # Model inference
    with torch.no_grad():
        logits = tone_model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map prediction to emotion
    emotion = id2label[str(predicted_class)]
    return emotion

def update_chunk_with_agent_data(idea_frame_pairs, chunk_id, agent, new_data):
    """
    Append new data to a specific chunk in the idea_frame_pairs without replacing existing data.
    """
    if 0 <= chunk_id < len(idea_frame_pairs):
        if agent not in idea_frame_pairs[chunk_id]:
            idea_frame_pairs[chunk_id][agent] = []
            print(f"Initializing {agent} for chunk {chunk_id + 1}.")
        
        # Ensure the new data is only appended once
        if new_data not in idea_frame_pairs[chunk_id][agent]:
            idea_frame_pairs[chunk_id][agent].append(new_data)
            print(f"Appended new data to {agent} for chunk {chunk_id + 1}: {new_data}")
        else:
            print(f"Data for {agent} already exists in chunk {chunk_id + 1}, skipping append.")
    else:
        print(f"Invalid chunk ID: {chunk_id}")



def tone_detection_agent(idea_frame_pairs, audio_path, chunk_idx=None):
    """
    Detect the tone of audio chunks and append the data to idea_frame_pairs.
    """
    if chunk_idx is not None:
        print(f"Processing chunk {chunk_idx + 1} for tone detection...")
        selected_chunk = [idea_frame_pairs[chunk_idx]]
    else:
        print("Processing all chunks for tone detection...")
        selected_chunk = idea_frame_pairs

    for idx, pair in enumerate(selected_chunk):
        # Ensure 'start' and 'end' exist
        start_time = pair.get("start")
        end_time = pair.get("end")

        if start_time is None or end_time is None:
            print(f"Skipping chunk {idx + 1} due to missing 'start' or 'end'.")
            continue

        # Perform tone analysis
        try:
            tone = analyze_tone(audio_path, start_time, end_time)
            print(f"Tone detected for chunk {idx + 1}: {tone}")
            update_chunk_with_agent_data(idea_frame_pairs, idx, "tone_analysis", tone)

        except Exception as e:
            print(f"Error processing chunk {idx + 1}: {e}")
            continue  # Skip this chunk and continue with the next


























