import streamlit as st
from streamlit_cropper import st_cropper
from methods import get_final_time, get_random_img, compute_matching_probs
import numpy as np
import datetime

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(layout="wide")

st.title("Welcome to the Spot cropper app")

label_list=["a cat", "a bunny","a penguin","an owl", 
            "an owl with a bow tie", "a house", "a person", "a raccoon", "junk"]

#text_tokens = compute_text_tokenization(label_list)

img, label = get_random_img()

if st.checkbox("Start Game!"):
    col1, col2 = st.columns([3,1])
    game_placeholder = col1.empty()
    #game_container = game_placeholder.container()
    now = datetime.datetime.now()
    #Wit cache resource, it is called just one time per session
    final_hour = get_final_time(total_seconds = 30)
    with game_placeholder.container():

        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                    aspect_ratio=(1, 1), should_resize_image=True)
        
    col2.write(f"Find the {label}")
    time_place = col2.empty()
    preview_text = col2.empty()
    preview_img =  col2.empty()
    final_label = col2.empty()
    success = False

    while now <= final_hour:
        time_place.header(str(round((final_hour-now).total_seconds())))

        # Manipulate cropped image at will
        preview_text.write("Preview")
            
        _ = cropped_img.thumbnail((150,150))
        preview_img.image(cropped_img)

        match_probs = compute_matching_probs(cropped_img, label_list)

        result = label_list[np.argmax(match_probs)]
        final_label.write(result)
        if result ==f"a {label}":
            success = True
            break

        now = datetime.datetime.now()

    get_final_time.clear()
    get_random_img.clear()

    if success:
        game_placeholder.success(f"You found the {label}!")
    else:
        game_placeholder.error(f"You didn't found the {label}")