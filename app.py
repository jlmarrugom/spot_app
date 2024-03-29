import streamlit as st
from streamlit_cropper import st_cropper
from methods import get_final_time, get_random_img, compute_matching_probs
import numpy as np
import datetime

st.set_page_config(layout="wide", page_icon=":mag:", page_title="Spot Finder")



#text_tokens = compute_text_tokenization(label_list)

img, label, label_list = get_random_img()
st.title(f"Where is the {label}? :mag:")

col1, col2 = st.columns([3,1])
game_placeholder = col1.empty()
#game_container = game_placeholder.container()
now = datetime.datetime.now()
#Wit cache resource, it is called just one time per session
final_hour = get_final_time(total_seconds = 30)
with game_placeholder.container():

    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=True, box_color="#D0312D",#'#0000FF',
                                aspect_ratio=(1, 1), should_resize_image=True)

if col1.button("Play again?"):
    get_final_time.clear()
    get_random_img.clear()
    st.experimental_rerun()
    
col2.subheader(f"Find the {label}")
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
    preview_img.image(cropped_img, width=256)

    match_probs = compute_matching_probs(cropped_img, label_list)

    result = label_list[np.argmax(match_probs)]
    final_label.write(result)
    if result ==f"a {label}":
        success = True
        break

    now = datetime.datetime.now()

if success:
    game_placeholder.success(f"You found the {label}!")
else:
    game_placeholder.error(f"You didn't found the {label}")