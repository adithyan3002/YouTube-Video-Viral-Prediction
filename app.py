import streamlit as st
import pickle
import numpy as np

# Load model
with open("yt_viral_model.pkl", "rb") as obj:
    model = pickle.load(obj)

st.title("🎬 YouTube Viral Video Predictor")

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url("data:image/png;base64,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");
#         background-size: cover;
#         background-position: center;
#         background-color: #000000;
#         background-repeat: no-repeat;
#         background-attachment: fixed;   
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

import base64
import os

def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function before UI starts
set_background("youu4.jpeg")


st.markdown("Enter your video details to predict if it'll go viral or not!")

# Inputs
view_count = st.number_input("View Count", min_value=0)
likes = st.number_input("Likes", min_value=0)
dislikes = st.number_input("Dislikes", min_value=0)
comment_count = st.number_input("Comment Count", min_value=0)

# Replace text input with multi-select for tags
all_tags = ['funny', 'vlog', 'tutorial', 'review', 'gameplay', 'music', 'challenge', 'reaction', 'tech', 'education']
selected_tags = st.multiselect("Select Tags", options=all_tags)

# Feature engineering (basic)
tag_count = len(selected_tags)

if st.button("Predict"):
    like_ratio = likes / (likes + dislikes + 1)  # +1 to avoid div by zero
    engagement = (likes + dislikes + comment_count) / (view_count + 1)

    input_data = np.array([[view_count, likes, dislikes, comment_count, tag_count, like_ratio, engagement]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🔥 Your video is predicted to go VIRAL!")
        st.markdown("**Great job!** This video is likely to perform well. Keep it up! 🚀")
        st.markdown("[Explore more viral ideas](https://www.youtube.com/feed/trending)")
    else:
        st.error("😞 This video might not go viral.")
        st.markdown("Don't worry! Failure is just feedback. Improve your tags or engagement. 💡")
        st.markdown("[Check trending content for inspiration](https://www.youtube.com/feed/trending)")
