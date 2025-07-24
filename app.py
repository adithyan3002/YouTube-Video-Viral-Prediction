import streamlit as st
import pickle
import numpy as np
import base64
import random
from datetime import date, datetime

# Load model
with open("yt_viral_video_model.pkl", "rb") as obj:
    model = pickle.load(obj)

def set_video_background(video_path):
    video_file = open(video_path, "rb").read()
    video_base64 = base64.b64encode(video_file).decode("utf-8")

    st.markdown(f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        .video-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
        }}
        .video-container video {{
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
            opacity: 0.2;
        }}
        .block-container {{
            position: relative;
            z-index: 2;
        }}
        </style>
        <div class="video-container">
            <video autoplay loop muted>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>
        """, unsafe_allow_html=True)


# Call this at the top of your script (after imports)
set_video_background("bg/youtube_background.mp4")

# Title---
st.title("\U0001F3AC YouTube Video Viral Predictor")
st.markdown("Enter your video details to predict if it'll go viral or not!")

# Input values
view_count = st.number_input("View Count", min_value=0)
likes = st.number_input("Like Count", min_value=0)
comment_count = st.number_input("Comment Count", min_value=0)

# Age input method
# input_method = st.radio("Select Time Unit", ["Hours", "Days", "Weeks", "Months"])
input_method = st.selectbox("Select Time Unit", ["Hours", "Days", "Weeks", "Months"])

if input_method == "Hours":
    age_hours = st.number_input("Video Age (in Hours)", min_value=0.0)
    video_age_days = age_hours / 24

elif input_method == "Days":
    age_days = st.number_input("Video Age (in Days)", min_value=0)
    video_age_days = age_days

elif input_method == "Weeks":
    age_weeks = st.number_input("Video Age (in Weeks)", min_value=0)
    video_age_days = age_weeks * 7

elif input_method == "Months":
    age_months = st.number_input("Video Age (in Months)", min_value=0)
    video_age_days = age_months * 30
else:
    st.error("Please select a valid time unit.")

# Feature engineering
likes_per_day = likes / (video_age_days + 1)

viral_comments =[
    "🔥 This content is on fire! Keep it up!",
    "🌟 You're on the right track to success!",
    "🚀 Amazing video – you're going viral for sure!",
    "💡 Bright idea! This one's a hit.",
    "🎯 You're hitting all the right marks!"
    "🚀 Your video is destined for greatness!",
    "🎯 Nailed it! Audiences will love this.",
    "🌟 You’ve hit the sweet spot — amazing job!",
    "📈 Trending vibes detected — you're going viral!",
    "👏 That’s the kind of content people can’t scroll past!",
    "💡 Creators like you change the game!",
    "🎬 Your story deserves to be seen!",
    "🎉 Let the success begin — this one's going places!",
    "🌍 You’re about to make waves!"
]

not_viral_comments = [
    "💪 Keep pushing, success is near!",
    "✨ Don’t give up – every video is a step forward.",
    "📈 Improve and you’ll go viral soon!",
    "🛠️ Try improving the tags or engagement next time.",
    "🌱 Every creator starts somewhere – keep growing!",
    "🔥 Don’t stop – your next video might be the big one!"
    "💪 Every big channel started somewhere — keep going!",
    "✨ Don’t give up — your content has potential.",
    "📈 Improvement is just one video away!",
    "🛠️ Great attempt! Keep refining your craft.",
    "🌱 Growth takes time. Stay consistent!",
    "🔁 Test, learn, and bounce back stronger!",
    "🧠 Smart creators learn from every upload.",
    "🎥 This is part of the journey. Next one might be the big one!",
    "🚧 Don’t fear low views — fear not creating.",
    "🏗️ You're building your success video by video!"
]

# Prediction
if st.button("Predict"):
    input_data = np.array([[view_count, likes, comment_count, video_age_days, likes_per_day]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("\U0001F525 Your video is predicted to go VIRAL!")
        st.balloons()
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
