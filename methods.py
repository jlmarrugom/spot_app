import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import datetime
import random
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
    img_file = os.path.join(assets_dir,random.choice(img_files))
    label_str = img_file.split("_")[-1].split(".")[0]
    if label_str=="bunny":
        label="bunny"
        label_list=["a cat", "a bunny"]
    elif label_str=="dog":
        label="dog"
        label_list = ["a dog", "a polar bear"]
    else:
        label="cat"
        label_list=["a cat","a penguin","an owl", 
            "an owl with a bow tie", "a house", "a person", "junk"]
    img = Image.open(img_file)
    return img, label, label_list

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