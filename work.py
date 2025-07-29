import streamlit as st 
import os
from mistralai import Mistral
import base64
from rag import ImageQuerySignature
import dspy 

api_key = 'AFENjNYLukhJROiwg5vvNKiZyV2T1Fq5'
client = Mistral(api_key=api_key)

if "state" not in st.session_state:
    st.session_state["state"] = None 

select = st.selectbox(label="choose the upload option", options=["camera", "upload"])

st.session_state["state"] = select 

image_model = dspy.ChainOfThought(signature=ImageQuerySignature)

if st.session_state["state"] == "camera":
    camera_input = st.camera_input(label="Take a shot of your question")
    if camera_input:
        # Read the image file from the UploadedFile object and encode it as base64
        image_bytes = camera_input.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_base64}"
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"{image_url}"
            }
        )
        text = None 
        for content in ocr_response.pages:
            text = content.markdown 
        user_query = st.text_input(label="Write your query")
        if st.button("Submit"):
            response = image_model(context = text, query = user_query).answer
            st.write(response)

if st.session_state["state"] == "upload":
    image_input = st.file_uploader(label="Upload the image", type=["jpg", "jpeg", "png"])
    if image_input:
        image_bytes = image_input.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_base64}"
        save_path = os.path.join(os.path.dirname(__file__), image_input.name)
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"{image_url}"
            }
        )
        text = None
        for content in ocr_response.pages:
            text = content.markdown
        st.write(text)
        user_query = st.text_input(label="Write your query")
        if st.button("Submit"):
            response = image_model(context = text, query = user_query).answer
            st.write(response)