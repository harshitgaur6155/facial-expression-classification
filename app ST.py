import streamlit as st
from PIL import Image
from facialExpressionClassify import logger
from facialExpressionClassify.utils.common import decodeImage, writeImage
from facialExpressionClassify.pipeline.predict import PredictionPipeline
import time
import io

class ClientApp:
    def __init__(self):
        self.classifier = None

    def classify_image(self, uploaded_file):

            pil_image = Image.open(uploaded_file)
            self.filename = "inputImage.jpg"
            pil_image.save(self.filename)
            
            self.classifier = PredictionPipeline(self.filename)

            result = self.classifier.streamlit_predict()
            return result

    

# Set up the page layout
st.set_page_config(page_title="Facial Expression Classification", layout="centered")

# Custom CSS for the page
st.markdown("""
    <style>
        body {
            background: url('https://img.freepik.com/free-photo/abstract-uv-ultraviolet-light-composition_23-2149243965.jpg?t=st=1739676960~exp=1739680560~hmac=786e0e7f3d58fb8d6cf6035bd6c39aa3c83c0e5e97043f72ba0a2ceb5fea0305&w=2000') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
        }

        h3 {
            font-size: 2.2rem;
            text-align: center;
            margin-bottom: 40px;
            font-weight: bold;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.6);
            background: -webkit-linear-gradient(45deg, #00bfff, #ff007f);
            -webkit-background-clip: text;
            color: transparent;
        }

        .main-card {
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 40px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.4);
            text-align: center;
            width: 100%;
            max-width: 900px;
        }

        .btn-upload {
            background: linear-gradient(45deg, #00bfff, #ff007f);
            border: none;
            padding: 15px;
            color: white;
            font-size: 1.2rem;
            border-radius: 20px;
            cursor: pointer;
            width: 80%;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .btn-upload:hover {
            background: linear-gradient(45deg, #007b9f, #e40062);
            transform: scale(1.05);
        }

        .loading {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            visibility: hidden;
            font-size: 1.5rem;
            color: white;
            font-weight: bold;
        }

        .loading.show {
            visibility: visible;
        }

        .prediction-result {
            display: none;
            font-size: 1.5rem;
            margin-top: 30px;
        }

        .gif-container {
            display: none;
            margin-top: 20px;
        }

        .gif-container img {
            max-width: 100%;
            height: auto;
            max-height: 230px;
            object-fit: contain;
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Upload Section
st.markdown('<h3>What’s Your Mood? Upload Your Image for a Quick Mood Check!</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "png", "jpeg"])

# Display uploaded image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Show loading spinner
    with st.spinner("Processing..."):

        # Initialize the ClientApp
        clApp = ClientApp()

        # Get the classification result from the classifier
        result = clApp.classify_image(uploaded_file)
        
        # Display the prediction result
        st.markdown(f"### Predicted Expression: {result.capitalize()}", unsafe_allow_html=True)

        # GIF based on prediction result
        gifs = {
            "anger": "anger.gif",
            "contempt": "contempt.gif",
            "disgust": "disgust.gif",
            "fear": "fear.gif",
            "happy": "happy.gif",
            "neutral": "neutral.gif",
            "sad": "sad.gif",
            "surprise": "surprise.gif"
        }

        # Find the corresponding GIF for the result
        gif_path = f"static/{gifs.get(result, 'neutral.gif')}"
        st.image(gif_path, use_container_width=True)