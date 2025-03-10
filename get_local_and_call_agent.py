##########TODO: 1) FIX UPDATING PART 2) additional captions seem to be th same as initial 3) fix face detector 2) make chatgpt loop, full loop with decision of continue etc and answering criteria 3) add in CLAP 4) add in prosody agent 

import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"  # Use the correct GPU ID


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
    # Import LLaVA model library AFTER setting the environment variable
    #import torch




# Run WhisperX to get sentence-level segments with timestamps
def extract_audio_transcript(video_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    
    device = "cpu"  # Change to "cuda" if you have a GPU
    model = whisperx.load_model("large-v2", device, compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    
    result = model.transcribe(audio, batch_size=1)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
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

# Group sentences into semantic chunks
def create_semantic_chunks(transcript_segments, threshold= 0.65): #0.65
    chunks = []
    current_chunk = {"text": transcript_segments[0]["text"], "start": transcript_segments[0]["start"], "end": transcript_segments[0]["end"]}
    current_embedding = get_bert_embedding(current_chunk["text"])

    for i in range(1, len(transcript_segments)):
        segment = transcript_segments[i]
        segment_embedding = get_bert_embedding(segment["text"])
        similarity = cosine_similarity(current_embedding, segment_embedding)

        if similarity >= threshold:
            current_chunk["text"] += " " + segment["text"]
            current_chunk["end"] = segment["end"]
        else:
            chunks.append(current_chunk)
            current_chunk = {"text": segment["text"], "start": segment["start"], "end": segment["end"]}
            current_embedding = segment_embedding

    chunks.append(current_chunk)
    print(f"Number of semantic chunks: {len(chunks)}")
    return chunks

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


# Process video to extract frames for each phrase segment
def process_video_for_llava(video_path):
    enable_gpu_for_llava()
    print(f"Extracting transcript from video: {video_path}")
    transcript_segments = extract_audio_transcript(video_path)
    
    print("Creating semantic chunks...")
    phrase_segments = create_semantic_chunks(transcript_segments)
    
    print("Extracting frames based on aligned phrases...")
    idea_frame_pairs = []

    for phrase_segment in phrase_segments:
        print(f"Extracting frames for phrase: '{phrase_segment['text']}'")
        print(f"Start time: {phrase_segment['start']}, End time: {phrase_segment['end']}")

        # Extract frames with every 4th frame selected based on the segment's time range
        segment_frames = load_video_frames(video_path, start_time=phrase_segment['start'], end_time=phrase_segment['end'], frame_skip=4)

        # If you want to further downsample or adjust the number of frames, you can add more logic here
        idea_frame_pairs.append({
            "text": phrase_segment['text'],
            "frames": segment_frames
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
        print(f"Processing chunk {idx}: {video_frames.shape}")

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
        print(f"Chunk {idx} caption: {caption}")
        update_chunk_with_agent_data(idea_frame_pairs, idx, "llov_caption", caption)
        captions[idx] = caption

    return captions
'''
def get_llov_caption(idea_frame_pairs, chunk_idx=None):
    enable_gpu_for_llava()
    if chunk_idx is None:
        chunk_indices = range(len(idea_frame_pairs))
    else:
        chunk_indices = [chunk_idx]

    captions = {}

    for idx in chunk_indices:
        video_frames = prep_for_llov(idea_frame_pairs, idx)
        print(f"Processing chunk {idx}: {video_frames.shape}")

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
        print(f"Chunk {idx} caption: {caption}")
        captions[idx] = caption

    return captions
'''


'''

def face_detection_chunk(idea_frame_pairs, chunk_idx=None):
    """
    Choose the chunk to process for face detection. If chunk_idx is specified, 
    process only that chunk; otherwise, process all chunks.
    """
    
    disable_gpu_for_face_detection()  # Disable GPU for face detection


    if chunk_idx is not None:
        print(f"Processing chunk {chunk_idx + 1} for face detection...")
        selected_chunk = [idea_frame_pairs[chunk_idx]]
        # Process selected chunk
        deep_face_detector.process_chunk_for_face_detection(selected_chunk, output_face_video_path)
    else:
        # Process all chunks
        print("Processing all chunks for face detection...")
        deep_face_detector.process_chunk_for_face_detection(idea_frame_pairs, output_face_video_path)




# Main function        
if __name__ == "__main__":
    video_path = "/home/itfelicia/chain/demo_6.mp4"
    audio_path = "/home/itfelicia/chain/demo_6_audio.wav" 
    output_face_video_path = "/home/itfelicia/chain/demo_6_analysed.mp4"

    # Process video to extract idea_frame_pairs
    idea_frame_pairs = process_video_for_llava(video_path)
    
    # Run face detection on a specific chunk 
    face_detection_chunk(idea_frame_pairs, 1)  #(idea_frame_pairs, [INDEX OF CHUNK YOU WANT TO RUN - PUT 4 TO RUN ON SEGMENT 5])
    
    # Run YOLO detection on a specific chunk 
    yolo_detector.yolo_detection_on_chunk(idea_frame_pairs, 4) #(idea_frame_pairs, [INDEX OF CHUNK YOU WANT TO RUN - PUT 4 TO RUN ON SEGMENT 5])

    print(f"Processed {len(idea_frame_pairs)} segments.")
    print("Processing segments through LLaVA OneVision...")
    
    llov_captions = get_llov_caption(idea_frame_pairs)
'''  

# Update the face detection function to return results
def face_detection_chunk(idea_frame_pairs, chunk_idx=None):
    """
    Choose the chunk to process for face detection. If chunk_idx is specified, 
    process only that chunk; otherwise, process all chunks.
    """
    disable_gpu_for_face_detection()  
    

    if chunk_idx is not None:
        print(f"Processing chunk {chunk_idx + 1} for face detection...")
        selected_chunk = [idea_frame_pairs[chunk_idx]]
        # Process selected chunk and return face detection results
        face_detection_results = deep_face_detector.process_chunk_for_face_detection(selected_chunk, output_face_video_path)
    else:
        # Process all chunks and return face detection results
        print("Processing all chunks for face detection...")
        face_detection_results = deep_face_detector.process_chunk_for_face_detection(idea_frame_pairs, output_face_video_path)

    return face_detection_results  # Return the face detection results







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
        video_frames = prep_for_llov(idea_frame_pairs, idx)
        if video_frames.size == 0:
            print(f"No frames available for chunk {idx + 1}. Skipping.")
            continue

        print(f"Processing chunk {idx + 1}: {video_frames.shape}")

        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\nGive an additional description of what's happening in this video, this time in even greater detail, especially the actions carried out."

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



def generate_prompt_from_video_data(idea_frame_pairs, llov_captions):
    """
    Generate a dynamic GPT prompt using processed video data.
    """
    prompt_parts = []
    for idx, pair in enumerate(idea_frame_pairs):
        start_time = pair.get('start', 'N/A')
        end_time = pair.get('end', 'N/A')
        audio_transcript = pair.get('text', 'No transcript available')
        llov_caption = llov_captions.get(idx, "No caption generated")
        yolo_detections = pair.get("yolo_detections", "No YOLO detections")

        # Append each chunk's details to the prompt
        prompt_parts.append(
            f"Chunk {idx + 1}: {start_time} - {end_time}\n"
            f"Audio Transcript: {audio_transcript}\n"
            f"LLaVA Caption: {llov_caption}\n"
            f"YOLO Detections: {yolo_detections}\n"
        )

    # Combine the structured data into a comprehensive video summary
    video_summary = "\n".join(prompt_parts)
    '''
    # Append task-specific instructions for GPT
    task_instructions = (
        "The two criteria for a video being classified as 'hateful' are:\n"
        "Criteria 1: Targets a certain individual or group of individuals based on a characteristic.\n"
        "Criteria 2: Discriminates against OR blames OR encourages harm/fear/hatred towards this individual/group "
        "OR threatens societal peace/harmony.\n\n"
        "Your goal is to assess whether these two criteria are met based on the content provided and choose an AGENT "
        "and a CHUNK ID to use that AGENT on that CHUNK to find out more information about the video. For the AGENTS, "
        "you can choose between:\n"
        "1) 'YOLO' to find out more about the objects in the video.\n"
        "2) 'LLOV' to find out more about the actions of the people in the video.\n"
        "3) 'DEEPFACE' to find out more about the characteristics of the people (race, gender, age, emotion).\n"
        "4) The audio events that happen.\n\n"
        "Answer format is: {Chunk ID, AGENT}"
    )
    '''
    
    
    ###TO ADD IN PROSODY AND FIXED DEEPFACE 
    
    # Append task-specific instructions for GPT
    task_instructions = (
        "The two criteria for a video being classified as 'hateful' are:\n"
        "Criteria 1: Targets a certain individual or group of individuals based on a characteristic.\n"
        "Criteria 2: Discriminates against OR blames OR encourages harm/fear/hatred towards this individual/group "
        "OR threatens societal peace/harmony.\n\n"
        "Your goal is to assess whether these two criteria are met based on the content provided and choose an AGENT "
        "and a CHUNK ID to use that AGENT on that CHUNK to find out more information about the video. For the AGENTS, "
        "you can choose between:\n"
        "1) 'YOLO' to find out more about the objects in the video.\n"
        "2) 'LLOV' to find out more about the actions of the people in the video.\n"
        "Answer format is: {Chunk ID, AGENT}"
    )


    # Combine the summary with task instructions
    return f"Video Summary:\n{video_summary}\n\n{task_instructions}"



def update_chunk_with_agent_data(idea_frame_pairs, chunk_id, agent, new_data):
    """
    Append new data to a specific chunk in the idea_frame_pairs without replacing existing data.
    """
    if 0 <= chunk_id < len(idea_frame_pairs):
        # Initialize the agent key as a list if not present
        if agent not in idea_frame_pairs[chunk_id]:
            idea_frame_pairs[chunk_id][agent] = []

        # Append the new data to the list for this agent
        idea_frame_pairs[chunk_id][agent].append(new_data)
        print(f"Appended {agent} data to chunk {chunk_id + 1}.")
    else:
        print(f"Invalid chunk ID: {chunk_id}")



###########################################TO ADD IN PROSODY 

def process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response):
    """
    Process GPT response to selectively run agents and append new information to idea_frame_pairs.
    """
    if not gpt_response:
        print("GPT response is None or empty. Skipping further processing.")
        return

    # Parse GPT response to identify which agents need to be run
    for line in gpt_response.splitlines():
        try:
            # Clean the line by removing curly braces, quotes, and extra spaces
            clean_line = line.strip().strip("{}").strip()  # Remove curly braces and extra spaces
            clean_line = clean_line.replace("'", "").replace('"', "")  # Remove single or double quotes

            # Split the cleaned line by comma
            chunk_id, agent = clean_line.split(",")
            chunk_id = int(chunk_id.strip()) - 1  # Adjust for 0-based indexing
            agent = agent.strip()

            # Run the specified agent on the chunk and append the data
            if agent == "LLOV":
                # Call llov_agent only when instructed by GPT
                llov_data = llov_agent(idea_frame_pairs, chunk_idx=chunk_id)  # Generate captions using LLOV
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "llov_caption", llov_data.get(chunk_id, "No data"))
            elif agent == "YOLO":
                yolo_data = yolo_detector.yolo_detection_on_chunk(idea_frame_pairs, chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "yolo_detections", yolo_data)
            elif agent == "DEEPFACE":
                deepface_data = face_detection_chunk(idea_frame_pairs, chunk_idx=chunk_id)
                update_chunk_with_agent_data(idea_frame_pairs, chunk_id, "deepface_analysis", deepface_data)
            else:
                print(f"Unknown agent: {agent}")
        except Exception as e:
            print(f"Error processing GPT response line '{line}': {e}")

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


if __name__ == "__main__":
    video_path = "/home/itfelicia/chain/demo_6.mp4"
    audio_path = "/home/itfelicia/chain/demo_6_audio.wav" 
    output_face_video_path = "/home/itfelicia/chain/demo_6_analysed.mp4"

    # Process video to extract idea_frame_pairs
    idea_frame_pairs = process_video_for_llava(video_path)

    print(f"Processed {len(idea_frame_pairs)} segments.")
    print("Processing segments through LLaVA OneVision...")
    
    # Get initial LLaVA captions
    llov_captions = get_llov_caption(idea_frame_pairs)
    
    # Generate the GPT prompt dynamically
    prompt = generate_prompt_from_video_data(idea_frame_pairs, llov_captions)
    print("Generated GPT Prompt:\n", prompt)

    # Send to GPT for analysis
    gpt_response = ask_company_chatgpt(prompt)
    print("Company GPT Response:\n", gpt_response)

    # Process GPT response and update idea_frame_pairs
    process_gpt_response_and_update_pairs(idea_frame_pairs, gpt_response)

    # Print updated summary
    print_updated_summary(idea_frame_pairs)
