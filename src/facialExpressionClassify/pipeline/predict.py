import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

# from transformers import cached_path # type: ignore
import requests
from io import BytesIO

import os
from facialExpressionClassify import logger
from facialExpressionClassify.constants import *
from facialExpressionClassify.utils.common import read_yaml



class PredictionPipeline:
    def __init__(self, filename, config_filepath = CONFIG_FILE_PATH):
        self.filename = filename
        self.config = read_yaml(config_filepath)


        # Load model (When Model is stored)
        model_path = Path(self.config.training.trained_model_path)


        # Download model file from Hugging Face
        model_url = "https://huggingface.co/harshitgaur6155/facial-expression-classification/resolve/main/custom_model_1.h5"

        # Download the model file
        response = requests.get(model_url)
        if response.status_code == 200:
            # Save the content to a local file
            with open(model_path, 'wb') as f:
                f.write(response.content)

        
        self.model = tf.keras.models.load_model(model_path)

        
        self.class_mapping = {
            0: 'anger',
            1: 'contempt',
            2: 'disgust',
            3: 'fear',
            4: 'happy',
            5: 'neutral',
            6: 'sad',
            7: 'surprise'
        }


    
    def predict(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Get prediction
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(result)

        # Map predicted index to class label
        prediction = self.class_mapping.get(result[0], "Unknown")

        return [{"image": prediction}]
    


    def streamlit_predict(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Get prediction
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(result)

        # Map predicted index to class label
        prediction = self.class_mapping.get(result[0], "Unknown")

        logger.info(f"Prediction: {prediction}")

        return prediction
