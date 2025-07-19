import streamlit as st
import pickle
import numpy as np
import base64
import os
import random

# Load model
with open("yt_viral_model.pkl", "rb") as obj:
    model = pickle.load(obj)

st.title("\U0001F3AC YouTube Video Viral Predictor")

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
            background-image-enhancement: brightness(0.8);
            background-image-quality: high;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set background image
set_background("bg\Youtube Logo Animation _ Youtube logo _ Animation of Youtube Logo _ Copyright Free.jpeg")

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

viral_comments =[
    "🔥 This content is on fire! Keep it up!",
    "🌟 You're on the right track to success!",
    "🚀 Amazing video – you're going viral for sure!",
    "👏 Audiences will love this!",
    "💡 Bright idea! This one's a hit.",
    "🎯 You're hitting all the right marks!"
]

not_viral_comments = [
    "💪 Keep pushing, success is near!",
    "✨ Don’t give up – every video is a step forward.",
    "📈 Improve and you’ll go viral soon!",
    "🛠️ Try improving the tags or engagement next time.",
    "🌱 Every creator starts somewhere – keep growing!",
    "🔥 Don’t stop – your next video might be the big one!"
]

if st.button("Predict"):
    like_ratio = likes / (likes + dislikes + 1)  # +1 to avoid div by zero
    engagement = (likes + dislikes + comment_count) / (view_count + 1)

    input_data = np.array([[view_count, likes, dislikes, comment_count, tag_count, like_ratio, engagement]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("\U0001F525 Your video is predicted to go VIRAL!")
        st.markdown("**Great job!** This video is likely to perform well. Keep it up! \U0001F680")
        st.markdown(random.sample(viral_comments, 3))
        st.markdown("Want to learn more about going viral?")
        st.markdown("[Explore more viral ideas](https://www.youtube.com/feed/trending)")
    else:
        st.error("\U0001F61E This video might not go viral.")
        st.markdown("Don't worry! Failure is just feedback. Improve your tags or engagement. \U0001F4A1")
        st.markdown(random.sample(not_viral_comments, 3))
        st.markdown("Need more inspiration? Check out trending content!")
        st.markdown("[Check trending content for inspiration](https://www.youtube.com/feed/trending)")
