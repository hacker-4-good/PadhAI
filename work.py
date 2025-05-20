import streamlit as st 
import os
from mistralai import Mistral
import base64

api_key = 'AFENjNYLukhJROiwg5vvNKiZyV2T1Fq5'
client = Mistral(api_key=api_key)

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
    for content in ocr_response.pages:
        st.header(content.markdown)