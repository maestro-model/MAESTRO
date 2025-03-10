###TODO: 1) sample 16 frames uniformly across video 2) extract answer 3) MAKE MORE ACCURATE?###


# -*- coding: utf-8 -*-


#generic imports 
import openai


def setup_openai_api():
    openai.api_key = config.COMPANY_API_KEY
    openai.api_type = config.API_TYPE
    openai.api_base = config.COMPANY_API_URL
    openai.api_version = config.API_VERSION
    
    
import time
import openai.error



from preprocess import transcribe_audio, segment_transcript_with_frames
import torch
import numpy as np
import config
import copy
from utils import sample_frames#, load_video, process_images
from operator import attrgetter
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu




#imports from llava repo 
import sys
import os
sys.path.append(os.path.abspath('/home/itfelicia/LLaVA-NeXT'))
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle



warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video with uniform sampling
def load_video(video_path, num_frames_to_sample):
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
        
    total_frame_num = len(vr)
    
    # Uniformly sample max_frames_num frames across the entire video length
    print(f"Sampling {num_frames_to_sample} across entire video")
    frame_indices = np.linspace(0, total_frame_num - 1, num_frames_to_sample, dtype=int)
    sampled_frames = vr.get_batch(frame_indices).asnumpy()
    
    return sampled_frames  # Shape: (num_frames_to_sample, height, width, channels)


# Load and process video with 8 sampled frames across the entire video
video_path = "/home/itfelicia/chain/demo_3.mp4"
video_frames = load_video(video_path, 16)  # This will sample 8 frames evenly across the video
print(video_frames.shape)  # Should be (16, height, width, channels)

image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\n Describe what is happening in this video in detail."


#f"{DEFAULT_IMAGE_TOKEN}\n What is happening in this video? The two criteria for a video being classified as 'hateful' is Criteria 1: Targets a certain individual or group of individuals based on a characteristic and Criteria 2: Discriminates against OR blames OR encourages harm/fear/hatred towards this individual/group of individuals OR threatens societal peace/harmony. Explain your answer for each criterion. Then, for each criterion, answer whether you need more information to determine if the video meets that criterion. For each, answer [Yes/No], and if [Yes], specify what additional information is required to answer the criteria (or [NIL] if no additional info is needed). It is important to answer that you need additional information if you are even a bit unsure.  For the additional info you need, you can choose between finding out more about 1) the objecs in the video 2) the actions of the people in the video 3) the characteristics of the people (race, gender, age, emotion) 4) the audio events that happen.

# f"{DEFAULT_IMAGE_TOKEN}\n What is happening in this video? The two criteria for a video being classified as 'hateful' is Criteria 1: Targets a certain individual or group of individuals based on a characteristic and Criteria 2: Discriminates against OR blames OR encourages harm/fear/hatred towards this individual/group of individuals OR threatens societal peace/harmony. Then, for each criterion, answer whether you need more information to determine if the video meets that criterion. For each, answer [Yes/No], and if [Yes], specify what additional information is required to answer the criteria (or [NIL] if no additional info is needed). For the additional info you need, you can choose between finding out more about 1) the objecs in the video 2) the actions of the people in the video 3) the hand gesture of the poeple 4) the facial expressions of the people It is important that you answer Yes (additional info is needed) if you are even a bit unsure. Answer format is [DESCRIPTION OF THE VIDEO], Need more info to answer criteria 1?: [YES/NO] & I need additional info about [WHAT TYPE OF ADDITIONAL INFORMATION YOU NEED TO ANSWER CRITERIA 1 / NIL if answer is NO], Need more info to answer criteria 2?: [YES/NO] & I need additional info about [WHAT TYPE OF ADDITIONAL INFORMATION YOU NEED TO ANSWER CRITERIA 2 / NIL if answer is NO], Prediction of hatefulness label: The video is [HATEFUL/NOT HATEFUL]"
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
print(text_outputs[0])
