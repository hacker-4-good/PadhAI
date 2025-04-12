import streamlit as st 
import os
from icrawler.builtin import GoogleImageCrawler
import nltk 
import time
from pytube import Search
from rag import ChatbotRAG
from io import BytesIO
from elevenlabs.client import ElevenLabs
from web_content import *

st.markdown(hero_logo, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(sidebar_logo, unsafe_allow_html=True)

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

google_Crawler = GoogleImageCrawler(storage = {'root_dir': 'Images'})

s1, s2 = st.columns([3,1])

if 'history' not in st.session_state:
    st.session_state.history = []

def get_response(question):
    response = ChatbotRAG().forward(question=question)
    return response

with s1:
    st.header("Ask Me 💭")
    ask_question = st.audio_input("Ask a question:")
    if ask_question:
        audio_data = BytesIO(ask_question.read())

        transcription = client.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1",
            tag_audio_events=True,
            language_code="eng", 
            diarize=True, 
        )
        question = transcription.text
        with st.spinner('Waiting for response...'):
            response = get_response(question)
            time.sleep(2)  
        st.session_state.history.append((question, response.answer))

    if st.session_state.history:
        for q, r in st.session_state.history:
            with st.chat_message("user"):
                st.write(f"**Question:** {q}")
            with st.chat_message("assistant"):
                st.write(f"**Answer:** {r}")

with s2:
    if st.session_state.history:
        st.header("Images")
        if st.session_state.history:
            q,r = st.session_state.history[-1]
            google_Crawler.crawl(keyword = f'show relevant Diagram or picture from NCERT textbook - Question: {q}, Answer: {r}', max_num = 5)
            image_folder = 'Images'
            if os.path.exists(image_folder):
                images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]
                if images:
                    with st.container(border=True, height=400):
                        st.image(images, caption=[os.path.basename(img) for img in images], use_container_width=True)
                for img in images:
                    os.remove(img)
        st.header("Video Explanation")
        if st.session_state.history:
            q,r = st.session_state.history[-1]
            try:
                search_video = Search(q)
                video = search_video.results[0]
                video_url = video.watch_url
                st.video(video_url)
            except Exception as e:
                st.error(f"Error Occurred: {e}")