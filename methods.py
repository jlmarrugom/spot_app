import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import datetime
from random import random
from PIL import Image
import os

@st.cache_resource
def get_final_time(total_seconds = 10):
    delta = datetime.timedelta(0,total_seconds)
    now = datetime.datetime.now()

    final_time = now + delta
    return final_time

@st.cache_resource
def get_random_img():
    # Upload an image and set some options for demo purposes
    assets_dir = "./assets/"
    img_files = os.listdir(assets_dir)
    img_file = os.path.join(assets_dir,img_files[round(random()*(len(img_files)-1))])
    label_str = img_file.split("_")[-1].split(".")[0]
    if label_str=="bunny":
        label="bunny"
    else:
        label="cat"
    img = Image.open(img_file)
    return img, label

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
    
def compute_matching_probs(cropped_img, label_list):
    model, preprocess = load_clip_model()
    inputs = preprocess(text=label_list, images=cropped_img, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).detach().numpy()
    
    return probs