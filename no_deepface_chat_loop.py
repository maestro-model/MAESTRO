import csv


import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use the correct GPU ID


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



import os;  os.environ["TF_USE_LEGACY_KERAS"] = "1"
import whisperx
import torch
import torch
torch.cuda.empty_cache()

import numpy as np
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import config
import sys
import os
import deep_face_detector
import warnings
import copy
import yolo_detector
from scipy.io import wavfile
import clap_detector


from audio_tone_detection import tone_detection_agent


# Add LLaVA path to sys path
sys.path.append(os.path.abspath('/home/itfelicia/LLaVA-NeXT'))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image

warnings.filterwarnings("ignore")

import torch

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN enabled: ", torch.backends.cudnn.enabled)



def disable_gpu_for_face_detection():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Import face detection library AFTER setting the environment variable
    #import cv2

def enable_gpu_for_llava():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # Import LLaVA model library AFTER setting the environment variable
    #import torch


# Run WhisperX to get sentence-level segments with timestamps
def extract_audio_transcript(video_path):
    clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", "_audio.wav")
    clip.audio.write_audiofile(audio_path)
    
    device = "cpu"  # Change to "cuda" if you have a GPU
    model = whisperx.load_model("large-v2", device, compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    
    result = model.transcribe(audio, batch_size=1, language="en")  # Explicitly set language to English
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)  # Force English alignment
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    transcript_segments = result["segments"]
    print(f"Number of segments found: {len(transcript_segments)}")
    return transcript_segments

# Initialize BERT model for semantic similarity
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings for sentences
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()

# Group sentences into semantic chunks and handle pauses separately
# Function to check if a pause segment contains significant sound
def check_pause_sound(audio, sample_rate, start_time, end_time, silence_threshold=1000):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    pause_audio = audio[start_sample:end_sample]
    amplitude = np.abs(pause_audio)
    if np.mean(amplitude) > silence_threshold:
        return True
    return False

# Group sentences into semantic chunks and handle pauses separately
def create_semantic_chunks_and_pauses(transcript_segments, audio_path, video_paththreshold=0.65):
    sample_rate, audio = wavfile.read(audio_path)
    
    chunks = []
    pauses = []
    current_chunk = {"text": transcript_segments[0]["text"], "start": transcript_segments[0]["start"], "end": transcript_segments[0]["end"]}
    current_embedding = get_bert_embedding(current_chunk["text"])

    for i in range(1, len(transcript_segments)):
        segment = transcript_segments[i]
        segment_embedding = get_bert_embedding(segment["text"])
        similarity = cosine_similarity(current_embedding, segment_embedding)

        pause_duration = segment["start"] - current_chunk["end"]

        if similarity >= threshold:
            current_chunk["text"] += " " + segment["text"]
            current_chunk["end"] = segment["end"]
        else:
            chunks.append(current_chunk)
            current_chunk = {"text": segment["text"], "start": segment["start"], "end": segment["end"]}
            current_embedding = segment_embedding

        # Handle pauses separately
        if pause_duration > 2:
            has_sound = check_pause_sound(audio, sample_rate, current_chunk["end"], segment["start"])
            if has_sound:
                pauses.append({"start": current_chunk["end"], "end": segment["start"], "duration": pause_duration})
    chunks.append(current_chunk)
    print(f"Number of semantic chunks: {len(chunks)}")
    print(f"Number of pauses: {len(pauses)}")
    return chunks, pauses

# Match frames to each aligned phrase based on timestamps
def load_video_frames(video_path, start_time, end_time, frame_skip=8):
    """
    Load frames from the video between start_time and end_time.
    This function will return every 4th frame within the time segment.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps

    # Convert times to frame indices
    start_frame = int((start_time / duration) * total_frames)
    end_frame = int((end_time / duration) * total_frames)
    
    # Calculate total number of frames to load based on the start and end times
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        return []

    # Select every 4th frame
    frame_indices = np.arange(start_frame, end_frame, frame_skip)
    
    # Ensure indices are within the valid range of total frames
    frame_indices = frame_indices[frame_indices < total_frames]

    frames = vr.get_batch(frame_indices).asnumpy()
    return frames
    
def extract_frames_for_segments(video_path, num_segments=4):
    """
    Splits the video into `num_segments` and extracts `frames_per_segment` evenly spaced frames from each segment.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    # Get the total duration of the video (in seconds)    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    total_duration = total_frames / fps
    
    # Calculate the start and end times for the four segments
    segment_durations = total_duration / 4
    segments = [
        {"start": 0, "end": segment_durations},
        {"start": segment_durations, "end": 2 * segment_durations},
        {"start": 2 * segment_durations, "end": 3 * segment_durations},
        {"start": 3 * segment_durations, "end": total_duration},
    ]

    print("Extracting frames for evenly split segments...")
    idea_frame_pairs = []

    for segment in segments:
        print(f"Extracting frames for segment: {segment['start']} to {segment['end']}")
        
        # Load the frames for the segment
        segment_frames = load_video_frames(
            video_path,
            start_time=segment["start"],
            end_time=segment["end"],
            frame_skip=4,
        )

        # Append segment data along with the frames
        idea_frame_pairs.append({
            "text": "",  
            "start": segment["start"],
            "end": segment["end"],
            "frames": segment_frames,
        })

    print(f"Processed {len(idea_frame_pairs)} segments with frames.")
    return idea_frame_pairs


# Process video to extract frames for each phrase segment
def process_video_for_llava(video_path, max_segments=None):
    """
    Process video to extract frames for each phrase segment, limited to max_segments if provided.
    """
    enable_gpu_for_llava()
    print(f"Extracting transcript from video: {video_path}")
    transcript_segments = extract_audio_transcript(video_path)

    if not transcript_segments:
        print(f"No audio detected in video {video_path}. Splitting into four segments with evenly spaced frames.")
        segments = extract_frames_for_segments(video_path, num_segments=4)
        return segments

    print("Creating semantic chunks...")
    
    audio_path = video_path.replace(".mp4", "_audio.wav")
    phrase_segments, pause_segments =  create_semantic_chunks_and_pauses(transcript_segments, audio_path)
    if pause_segments:
        results = clap_detector.process_pauses(pause_segments, audio_path)
        for result in results:
            if result:
                print(f"Top label: {result[0]}, Score: {result[1]}")
            else:
                print("No relevant result for this pause.")

    
    if not phrase_segments:
        print(f"No semantic chunks created for video {video_path}. Skipping processing.")
        return []
    print("Extracting frames based on aligned phrases...")
    idea_frame_pairs = []

    for phrase_segment in phrase_segments:
        print(f"Extracting frames for phrase: '{phrase_segment['text']}'")
        print(f"Start time: {phrase_segment['start']}, End time: {phrase_segment['end']}")

        segment_frames = load_video_frames(
            video_path,
            start_time=phrase_segment["start"],
            end_time=phrase_segment["end"],
            frame_skip=4,
        )

        idea_frame_pairs.append({
            "text": phrase_segment["text"],
            "start": phrase_segment["start"],
            "end": phrase_segment["end"],
            "frames": segment_frames,
        })

    print(f"Processed {len(idea_frame_pairs)} phrase segments with frames.")
    return idea_frame_pairs


    
# Set up LLaVA OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"#"cuda"
device_map = "auto" #CANNOT SET AS DEVICE_MAP = DEVICE 
llava_model_args = {"multimodal": True}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)
model.eval()


def prep_for_llov(idea_frame_pairs, chunk_idx,max_frames_num=5):
    
    
    if chunk_idx is not None:
        print(f"prepping chunk {chunk_idx + 1} for llov")
        selected_chunk = [idea_frame_pairs[chunk_idx]]
    else:
        print("prepping all chunks for llov")
        selected_chunk = idea_frame_pairs

    all_frames = []  
    for idx, pair in enumerate(selected_chunk):
        segment_text = pair["text"]
        frames = [frame.astype(np.uint8) for frame in pair["frames"] if frame is not None]
        
        # If no frames are available, skip this chunk
        if not frames:
            print(f"Skipping segment {idx + 1}: No valid frames available.")
            pair["yolo_detections"] = "No frames available"
            continue
        
        print(f"Processing Segment: {segment_text}")
        
        # Sampling frames uniformly
        total_frame_num = len(frames)
        if total_frame_num > max_frames_num:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frames = [frames[i] for i in uniform_sampled_frames]

        # Stack frames into a single NumPy array
        stacked_frames = np.stack(frames, axis=0)  # Shape: (frames, height, width, channels)
        all_frames.append(stacked_frames)
    
    # Concatenate all the chunks' frames if needed (e.g., stacking frames across chunks)
    if all_frames:
        final_frames = np.concatenate(all_frames, axis=0)  # Shape: (total_frames, height, width, channels)
        return final_frames
    else:
        return np.array([])  


def get_llov_caption(idea_frame_pairs, chunk_idx=None):
    enable_gpu_for_llava()
    if chunk_idx is None:
        chunk_indices = range(len(idea_frame_pairs))
    else:
        chunk_indices = [chunk_idx]

    captions = {}

    for idx in chunk_indices:
        video_frames = prep_for_llov(idea_frame_pairs, idx)
        print(f"Processing segment {idx+1}: {video_frames.shape}")

        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        caption = text_outputs[0]
        print(f"Segment {idx+1} caption: {caption}")
        update_chunk_with_agent_data(idea_frame_pairs, idx, "llov_caption", caption)
        captions[idx] = caption

    return captions


# Update the face detection function to return results
import tensorflow as tf

def face_detection_chunk(video_path, idea_frame_pairs, chunk_idx=None ):
    """
    Choose the chunk to process for face detection. If chunk_idx is specified, 
    process only that chunk; otherwise, process all chunks.
    """
    import os
    original_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")  # Save the original value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for this function
    
    try:
        print("GPU disabled for face detection.")
        
        # Force TensorFlow to use the CPU
        with tf.device('/CPU:0'):
            if chunk_idx is not None:
                print(f"Processing chunk {chunk_idx + 1} for face detection...")
                selected_chunk = [idea_frame_pairs[chunk_idx]]
                output_face_video_path = video_path.replace(".mp4", "_analysed.mp4")
                # Process selected chunk and return face detection results
                deepface_result = deep_face_detector.process_chunk_for_face_detection(selected_chunk, output_face_video_path)
                # Update the specific chunk in idea_frame_pairs
                #idea_frame_pairs[chunk_idx]["deepface_analysis"] = deepface_result
            else:
                # Process all chunks and return face detection results
                print("Processing all chunks for face detection...")
                deepface_result = deep_face_detector.process_chunk_for_face_detection(idea_frame_pairs, output_face_video_path)
                # Update all chunks in idea_frame_pairs
                #for i, chunk_info in enumerate(deepface_result):
                    #idea_frame_pairs[i]["deepface_analysis"] = chunk_info

            return deepface_result  # Return the face detection results
    except Exception as e:
        print(f"Error during face detection: {e}")
    finally:
        # Restore the original CUDA_VISIBLE_DEVICES setting
        if original_cuda_env is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_env
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]  # Remove the variable if it was not set
        print("GPU settings restored.")



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


##################################################

def llov_agent(idea_frame_pairs, chunk_idx=None):
    """
    Generate additional detailed captions using LLaVA for the specified chunks.
    """
    enable_gpu_for_llava()
    if chunk_idx is None:
        chunk_indices = range(len(idea_frame_pairs))
    else:
        chunk_indices = [chunk_idx]

    captions = {}

    for idx in chunk_indices:
        video_frames = prep_for_llov(idea_frame_pairs, idx, max_frames_num=8)
        if video_frames.size == 0:
            print(f"No frames available for chunk {idx + 1}. Skipping.")
            continue

        print(f"Processing chunk {idx + 1}: {video_frames.shape}")

        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n Give a DIFFERENT description of this part of the video. It is VERY IMPORTANT that you DO NOT REPEAT past descriptions. You can consider - What emotions are being conveyed by the characters in the video? How do their facial expressions and body language reflect their feelings?"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.3,##TO EXPERIMENT!!!!!!!!!!!!!!!!!!, 0.7 gives wrong conclusion
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        caption = text_outputs[0]
        print(f"Chunk {idx + 1} detailed caption: {caption}")
        captions[idx] = caption

        # Append the new caption to the chunk's data
        update_chunk_with_agent_data(idea_frame_pairs, idx, "llov_caption", caption)

    return captions




import openai
import time
import tiktoken  # Make sure to install this library
import pytesseract
from PIL import Image

company_api_key = "981bdca6dae44b78a930541b4577f696"
company_api_url = "https://dso-ie-openai.openai.azure.com/"

# Ensure the company's ChatGPT API key is used directly
openai.api_key = company_api_key

def estimate_cost(query, context, output, model='gpt-4'):
    '''
    Estimate cost of your query to GPT4
    '''
    cost_dict = {'gpt-4': {
                    'input': 0.03, 
                    'output': 0.06
                 },
                 'gpt-3.5': {
                    'input': 0.001,
                    'output': 0.002 
                 }
                }
    enc = tiktoken.encoding_for_model(model)
    
    # Ensure context, query, and output are strings
    context_str = str(context)
    query_str = str(query)
    output_str = str(output)

    cost = ((len(enc.encode(context_str + query_str)) / 1000) * cost_dict[model]['input']) + ((len(enc.encode(output_str)) / 1000) * cost_dict[model]['output'])

    print('Cost of query: USD$ {}'.format(round(cost, 2)))


def ask_company_chatgpt(prompt, model='gpt-4', max_tokens=256):
    try:
        # Call the company's ChatGPT API
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base = "https://dso-ie-openai.openai.azure.com/"
        openai.api_key = "981bdca6dae44b78a930541b4577f696"

        if model == 'gpt-4':
            azure_oai_model = "dsogpt4"
        else:
            azure_oai_model = "dsochatgpt35"

        response = openai.ChatCompletion.create(
            engine=azure_oai_model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": prompt}
            ]
        )

        output = response.choices[0].message.content
        if output:
            return output
        else:
            print("GPT returned an empty response.")
            return None

    except Exception as e:
        print(f"Error during GPT API call: {e}")
        return None




###########################################TO ADD IN PROSODY 
def process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response, video_path):
    """
    Process GPT response to selectively run agents and append new information to idea_frame_pairs.
    Supports responses in the format: {CONTINUE, Chunk ID, AGENT} or {CONTINUE, Chunk <ID>, AGENT}.
    """
    if not gpt_response:
        print("GPT response is None or empty. Skipping further processing.")
        return

    # Parse GPT response line-by-line
    for line in gpt_response.splitlines():
        try:
            # Clean the line and strip any wrapping symbols like "{}"
            clean_line = line.strip().strip("{}").replace("'", "").replace('"', "")
            print(f"Processing GPT response line: {clean_line}")

            # Split into components
            parts = clean_line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Expected three parts (CONTINUE, Chunk ID, AGENT), got: {parts}")

            # Extract and normalize action, chunk ID, and agent
            action = parts[0].strip().upper()
            chunk_id_raw = parts[1].strip().lower()  # To handle prefixes like "chunk"
            agent = parts[2].strip().upper()

            # Ensure action is "CONTINUE"
            if action != "CONTINUE":
                print(f"Unexpected action: {action}. Skipping line.")
                continue

            # Handle "Chunk" prefix in the chunk ID
            if "chunk" in chunk_id_raw:
                chunk_id_raw = chunk_id_raw.replace("chunk", "").strip()

            # Convert chunk ID to integer (1-based to 0-based indexing)
            chunk_id = int(chunk_id_raw) - 1
            print(f"Parsed Action: {action}, Chunk ID: {chunk_id + 1}, Agent: {agent}")

            # Call the appropriate agent
            if agent == "LLOV":
                llov_data = llov_agent(idea_frame_pairs, chunk_idx=chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "llov_caption", llov_data.get(chunk_id, "No data"))
            elif agent == "YOLO":
                yolo_data = yolo_detector.yolo_detection_on_chunk(idea_frame_pairs, chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "yolo_detections", yolo_data)
            elif agent == "DEEPFACE":
                deepface_data = face_detection_chunk(video_path, idea_frame_pairs, chunk_idx=chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "deepface_analysis", deepface_data)
            elif agent == "TONE":
                tone_data = tone_detection_agent(idea_frame_pairs, audio_path, chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "tone_analysis", tone_data)

            else:
                raise ValueError(f"Unknown agent: {agent}")

        except ValueError as ve:
            print(f"Invalid GPT response line format: '{line}'. Error: {ve}")
        except Exception as e:
            print(f"Unexpected error processing GPT response line '{line}': {e}")


def print_updated_summary(idea_frame_pairs):
    """
    Print the updated summary including all accumulated data for each chunk.
    """
    print("PRINTING UPDATED SUMMARY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for idx, pair in enumerate(idea_frame_pairs):
        print(f"Segment {idx + 1}:")
        print(f"  Audio transcript: {pair.get('text', 'No transcript')}")
        
        # Print all YOLO detections
        yolo_detections = pair.get('yolo_detections', [])
        if yolo_detections:
            print(f"  YOLO Detections:")
            for detection in yolo_detections:
                print(f"    - {detection}")
        else:
            print("  YOLO Detections: No detections")

        # Print all LLaVA captions (including past and present)
        llov_captions = pair.get('llov_caption', [])
        if llov_captions:
            print(f"  LLaVA Captions:")
            for caption in llov_captions:
                print(f"    - {caption}")
        else:
            print("  LLaVA Captions: No captions generated")

        # Print all DeepFace analysis results
        deepface_analysis = pair.get('deepface_analysis', [])
        if deepface_analysis:
            print(f"  DeepFace Analysis:")
            for analysis in deepface_analysis:
                print(f"    - {analysis}")
        else:
            print("  DeepFace Analysis: No analysis")
            
                # Include Tone Analysis
        tone_analysis = pair.get("tone_analysis", [])
        if tone_analysis:
            segment_summary += f"  Tone Analysis:\n"
            for tone in tone_analysis:
                segment_summary += f"    - {tone}\n"
        else:
            segment_summary += "  Tone Analysis: No tone detected\n"
        
        




def generate_prompt_from_video_data(idea_frame_pairs):
    """
    Generate a dynamic GPT prompt using the most recent processed video data.
    This function ensures that updated data like YOLO detections, LLaVA captions, and DeepFace analysis
    are included in the prompt for each segment.
    """
    prompt_parts = []
    
    # Loop through each segment and include the necessary data
    for idx, pair in enumerate(idea_frame_pairs):
        start_time = pair.get('start', 'N/A')
        end_time = pair.get('end', 'N/A')
        audio_transcript = pair.get('text', 'No transcript available')

        # Fetch the updated data for each segment
        yolo_detections = pair.get('yolo_detections', [])
        llov_captions = pair.get('llov_caption', [])
        deepface_analysis = pair.get('deepface_analysis', [])
        tone_analysis = pair.get("tone_analysis", [])
        # Construct the segment summary
        segment_summary = f"Segment {idx + 1} ({start_time} - {end_time}):\n"
        segment_summary += f"  Audio Transcript: {audio_transcript}\n"
        
        # Include YOLO detections
        if yolo_detections:
            segment_summary += f"  YOLO Detections:\n"
            for detection in yolo_detections:
                segment_summary += f"    - {detection}\n"
        else:
            segment_summary += "  YOLO Detections: No detections\n"
        
        # Include LLaVA captions
        if llov_captions:
            segment_summary += f"  LLaVA Captions:\n"
            for caption in llov_captions:
                segment_summary += f"    - {caption}\n"
        else:
            segment_summary += "  LLaVA Captions: No captions generated\n"
        
        # Include DeepFace analysis
        if deepface_analysis:
            segment_summary += f"  DeepFace Analysis:\n"
            for analysis in deepface_analysis:
                segment_summary += f"    - {analysis}\n"     
        else:
            segment_summary += "  DeepFace Analysis: No analysis\n"
          
        #include tone analysis          
        if tone_analysis:
            segment_summary += f"  Tone Analysis:\n"
            for tone in tone_analysis:
                segment_summary += f"    - {tone}\n"
        else:
            segment_summary += "  Tone Analysis: No tone detected\n"
        

       

        # Append the constructed segment summary to the full prompt
        prompt_parts.append(segment_summary)
    
    # Combine all segment summaries into one final video summary
    video_summary = "\n".join(prompt_parts)
    return video_summary


def generate_loop_prompt(idea_frame_pairs, round_number, force_continue=False):
    """
    Generate a prompt for ChatGPT to decide whether to continue or stop processing.
    Uses the most updated video data.
    """
    # Construct the video summary using the most up-to-date `idea_frame_pairs`
    video_summary = generate_prompt_from_video_data(idea_frame_pairs)

    # First-round instructions: force "CONTINUE"
    if force_continue:
        additional_instructions = (
            "This is the first round of analysis, and you must forcefully CONTINUE searching for information.\n"
            "Choose an AGENT and a CHUNK to focus on and answer in the format: {CONTINUE, Chunk ID, AGENT}.\n"
            "you can ONLY choose between :\n"
            "1) 'YOLO' to find out more about the objects in the video.\n"
            "2) 'LLOV' to find out more about the actions of the people in the video.\n"
            "3) 'TONE' to find out more about the tone of the audio in that segment.\n"
        )
    else:
        # Subsequent rounds: allow ChatGPT to infer hatefulness or decide to continue
        additional_instructions = (
            "Assess whether the video meets any of the following two criteria. **Meeting any one of these criteria** is sufficient to classify the video as OFFENSIVE or NORMAL. "
            "Your goal is to assess whether the content provided meets **either** of these criteria, and if so, classify the video as NORMAL or OFFENSIVE:\n"
            
            "Criteria 1: The video or its content targets a specific individual or group based on a characteristic, such as race, gender, appearance, etc.\n"
            
            "Criteria 2: The video contains generalizations, discrimination, inappropriate statements, or blame directed at these individuals/groups. It may also imply or encourage harm, fear, hatred, or societal disruption towards them, even if it doesn't explicitly advocate violence.\n"
            
            "If **either** of these criteria is met, classify the video as **OFFENSIVE**. IF THE VIDEO IS EVEN SLIGHTLY OFFENSIVE PLS CLASSIFY IT AS OFFENSIVE. If both criteria are met, the video should still be classified as OFFENSIVE. Please provide an explanation for your decision.\n"
            
            "Your task is to assess whether the video meets these criteria based on the content provided. If you believe the video is OFFENSIVE based on the criteria, answer in the format:\n"
            "{STOP, Explanation on whether video is OFFENSIVE or not}\n"
            
            "If you believe you need more information to determine whether the video is OFFENSIVE, choose an agent and a chunk ID to gather more data. Answer in the format:\n"
            "{CONTINUE, Chunk ID, AGENT}\n"
            
            "For the AGENTS, you can ONLY choose between:\n"
            "1) 'YOLO' to find out more about the objects present in the video in that segment.\n"
            "2) 'LLOV' to find out more about the actions of the people in that segment.\n"
            "3) 'TONE' to find out more about the tone of the audio in that segment.\n"
        )


    # Combine summary and instructions
    return (
        f"Round: {round_number}\n"
        f"Video Summary:\n{video_summary}\n\n"
        f"{additional_instructions}"
    )



def check_hatefulness(final_response):
    """Check if the first sentence in the final response indicates the video is hateful or not hateful."""
    # Split the response into sentences by looking for common sentence-ending punctuation (period, exclamation mark, question mark)
    sentence_endings = ['.', '!', '?']
    first_sentence = final_response.strip()

    # Find the position of the first sentence-ending punctuation
    for punctuation in sentence_endings:
        if punctuation in first_sentence:
            first_sentence = first_sentence.split(punctuation)[0].strip()
            break

    # Convert the first sentence to lowercase for comparison
    first_sentence_lower = first_sentence.lower()

    # Check if the sentence contains "hateful" and "not" together
    if "hateful" in first_sentence_lower and "not" in first_sentence_lower:
        return "Normal"  # Change "Not Hateful" to "Normal"
    
    # Check if the sentence contains "hateful" without "not"
    elif "hateful" in first_sentence_lower:
        return "Hateful"
    
    # If neither condition is met, classify as Unknown
    else:
        return "Unknown"


def save_result_to_tsv(video_id, final_response, hatefulness, output_tsv):
    """Append a single video ID, final response, and hatefulness classification to the TSV file."""
    with open(output_tsv, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow([video_id, final_response, hatefulness])  # Write all required fields




def process_video_files_in_folder(folder_path, output_tsv):
    """Process all videos in the given folder and save the final responses and classifications to a TSV."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            video_id = os.path.splitext(filename)[0]
            print(f"Processing video: {video_path}")

            try:
                idea_frame_pairs = process_video_for_llava(video_path)
                print(f"Processed {len(idea_frame_pairs)} segments.")
                llov_captions = get_llov_caption(idea_frame_pairs)

                round_number = 1
                continue_searching = True

                while continue_searching:
                    force_continue = (round_number == 1)
                    loop_prompt = generate_loop_prompt(idea_frame_pairs, round_number, force_continue=force_continue)
                    print(f"Generated Prompt for Round {round_number}:\n{loop_prompt}")
                    gpt_response = ask_company_chatgpt(loop_prompt)
                    print(f"Round {round_number} GPT Response:\n{gpt_response}")
                    minimum_rounds = 2
                    if round_number < minimum_rounds:
                        continue_searching = True

                    if "STOP" in gpt_response.upper() or round_number == 5:
                        continue_searching = False
                        final_video_summary = generate_prompt_from_video_data(idea_frame_pairs)
                        final_question = (
                            f"The two criteria for a video being classified as 'OFFENSIVE' are:\n"
                            "Criteria 1: ...\nCriteria 2: ...\n\n"
                            f"Video Summary:\n{final_video_summary}\n\"n"
                            f"Based on the provided information, is the video OFFENSIVE? Answer: YES/NO and explain."
                        )
                        final_answer = ask_company_chatgpt(final_question)
                        print("Final Hatefulness Determination:\n", final_answer)
                        hatefulness = check_hatefulness(final_answer)
                        save_result_to_tsv(video_id, final_answer, hatefulness, output_tsv)
                    elif "CONTINUE" in gpt_response.upper():
                        process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response, video_path)
                        round_number += 1
                        save_result_to_tsv(video_id, f"Round {round_number} GPT Response: {gpt_response}", "In Progress", output_tsv)
                    else:
                        print("Unexpected GPT response. Ending loop.")
                        break

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                
def process_single_video(video_path):
    """Process a single video and save the final responses and classifications to a TSV."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {video_path}")

    try:
        idea_frame_pairs = process_video_for_llava(video_path)
        print(f"Processed {len(idea_frame_pairs)} segments.")
        llov_captions = get_llov_caption(idea_frame_pairs)

        round_number = 1
        continue_searching = True

        while continue_searching:
            force_continue = (round_number == 1)
            loop_prompt = generate_loop_prompt(idea_frame_pairs, round_number, force_continue=force_continue)
            print(f"Generated Prompt for Round {round_number}:\n{loop_prompt}")
            gpt_response = ask_company_chatgpt(loop_prompt)
            print(f"Round {round_number} GPT Response:\n{gpt_response}")
            minimum_rounds = 2
            if round_number < minimum_rounds:
                continue_searching = True

            if "STOP" in gpt_response.upper() or round_number == 5:
                continue_searching = False
                final_video_summary = generate_prompt_from_video_data(idea_frame_pairs)
                final_question = (
                    f"The two criteria for a video being classified as 'OFFENSIVE' are:\n"
                    "Criteria 1: ...\nCriteria 2: ...\n\n"
                    f"Video Summary:\n{final_video_summary}\n\n"
                    f"Based on the provided information, is the video OFFENSIVE? Answer: YES/NO and explain."
                )
                final_answer = ask_company_chatgpt(final_question)
                print("Final Hatefulness Determination:\n", final_answer)
                hatefulness = check_hatefulness(final_answer)
            elif "CONTINUE" in gpt_response.upper():
                process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response, video_path)
                round_number += 1
            else:
                print("Unexpected GPT response. Ending loop.")
                break

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")

def process_video_files_in_list(folder_path, video_ids, output_tsv):
    """Process the videos from the given list of video IDs and save the final responses and classifications to a TSV."""
    for video_id in video_ids:
        video_path = os.path.join(folder_path, f"{video_id}.mp4")
        print(f"Processing video: {video_path}")

        try:
            idea_frame_pairs = process_video_for_llava(video_path)
            print(f"Processed {len(idea_frame_pairs)} segments.")
            llov_captions = get_llov_caption(idea_frame_pairs)

            round_number = 1
            continue_searching = True

            while continue_searching:
                force_continue = (round_number == 1)
                loop_prompt = generate_loop_prompt(idea_frame_pairs, round_number, force_continue=force_continue)
                print(f"Generated Prompt for Round {round_number}:\n{loop_prompt}")
                gpt_response = ask_company_chatgpt(loop_prompt)
                print(f"Round {round_number} GPT Response:\n{gpt_response}")
                minimum_rounds = 2
                if round_number < minimum_rounds:
                    continue_searching = True

                if "STOP" in gpt_response.upper() or round_number == 5:
                    continue_searching = False
                    final_video_summary = generate_prompt_from_video_data(idea_frame_pairs)
                    final_question = (
                        f"The two criteria for a video being classified as 'OFFENSIVE' are:\n"
                        "Criteria 1: ...\nCriteria 2: ...\n\n"
                        f"Video Summary:\n{final_video_summary}\n\n"
                        f"Based on the provided information, is the video OFFENSIVE? Answer: YES/NO and explain."
                    )
                    final_answer = ask_company_chatgpt(final_question)
                    print("Final Hatefulness Determination:\n", final_answer)
                    hatefulness = check_hatefulness(final_answer)
                    save_result_to_tsv(video_id, final_answer, hatefulness, output_tsv)
                elif "CONTINUE" in gpt_response.upper():
                    process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response, video_path)
                    round_number += 1
                    save_result_to_tsv(video_id, f"Round {round_number} GPT Response: {gpt_response}", "In Progress", output_tsv)
                else:
                    print("Unexpected GPT response. Ending loop.")
                    break

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")


if __name__ == "__main__":
    folder_path = "/home/itfelicia/MultiHateClip/videos/"  # Path to your video folder
    output_tsv = "/home/itfelicia/MultiHateClip/no_deepface.tsv"  # Path to save the results
    video_ids = [ "UeRqjzlI4jY", "k9OtaMbK0Ac", "_qldaPBgkk0", "R3Xt1__7TwQ", "uPJtlBAOT_U", "qLa59rRqxaE", "FuXd3OYtlQU", "lzKJ_AWegCc", "CdeJBx1r6qY", "RAGvPqMoW-4", "ulze2hXRwJg", "cXRgVEENkPA", "ndqGWFON2Dk", "01ygFLVdj8s", "jEdH8-AZ3aU", "LMOHDiG5XQg","C8VGhCg3gE0", "TvuSOkN7OrM", "lnsoi9nh8LI", "yEgFUrFnc4k", "ogsMoRZXFuk", "1HYp_bfklBI", "5aBoXKjdSXE", "VCLZPcfJrjc", "d1s_eLvyEOY", "8zLoOqXvk64", "pOV_1-vWVrE", "KDcCiUU8q5E", "cYQyH7hbNnw", "XlJCNPi5inM", "CChPIZ7vOzY", "gCIk9MfWM44", "YdRuekAOX-Y", "j6tCb7d_py4", "j_foVftOOs4", "WYJ3_pvq0aw", "zEBmhb3qikQ", "us7_-V0ufC4", "vQ9wQego_8A", "wq2cJV5Bz_0", "UEsC4gjQUMw", "hYyxxngdwRY", "c_7Hwq9E0NE", "8pEa6Jfz8bA", "m4SVUBK5Vyc", "sYpThqzEIVE", "pVGBplnEhl4", "35ptr5z4bOQ", "JlRVUempEB0", "eEbGvqzYVxU", "2-oUaI3Jb2c", "lUSBN2p04iA", "T9OlXnaLiTo", "gnDhz9NQP7s", "8E3cYSx7-iw", "Xw47KThr2ZQ", "6tT_NoDOOsQ"]

    process_video_files_in_list(folder_path, video_ids, output_tsv)
