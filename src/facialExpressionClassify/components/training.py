import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from facialExpressionClassify import logger
from facialExpressionClassify.entity.config_entity import ModelTrainingConfig
import collections
import numpy as np
from sklearn.utils.class_weight import compute_class_weight # type: ignore
# tf.config.run_functions_eagerly(True)



class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
    
    

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    
    
    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            # class_mode="sparse",  ## When to use Label (Integer) Encoding
            class_mode="categorical" ## Default in params --> When to use One-Hot Encoding of the classes.
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                # rotation_range=40,
                # horizontal_flip=True,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
                # shear_range=0.2,
                # zoom_range=0.2,
                # **datagenerator_kwargs

                
                rotation_range=15,  # Reduced
                horizontal_flip=True,
                width_shift_range=0.1,  
                height_shift_range=0.1,  
                shear_range=0.1,  
                zoom_range=0.1,  
                fill_mode="nearest",
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # for i, (batch_images, batch_labels) in enumerate(self.train_generator):
        #     logger.info(f"Batch {i+1}: Images Shape: {batch_images.shape}")
        #     logger.info(f"Batch {i+1}: Labels: {batch_labels}")

        #     # Stop after one full epoch to avoid an infinite loop
        #     if i >= len(self.train_generator):
        #         break

        counter = collections.Counter(self.train_generator.classes)
        logger.info(f"Class distribution: {counter}")
        logger.info(f"Train Generator Batch Size: {self.train_generator.batch_size}")
        logger.info(f"Validation Generator Batch Size: {self.valid_generator.batch_size}")

        # **NEW CODE**: Compute class weights
        class_labels = np.unique(self.train_generator.classes)  # Get unique class labels
        class_samples = [list(self.train_generator.classes).count(i) for i in class_labels]  # Count samples per class

        class_weights = compute_class_weight(
            'balanced', 
            classes=class_labels, 
            y=np.concatenate([np.full(n, i) for i, n in enumerate(class_samples)])
        )
        self.class_weights_dict = dict(zip(class_labels, class_weights))  # Create class weight dictionary

        logger.info(f"Class weights: {self.class_weights_dict}")  # Log class weights


    
    
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Convert non-serializable attributes before saving the model."""
        for layer in model.layers:
            layer_dict = layer.__dict__.copy()  # Create a copy of the dictionary
            for attr, value in layer_dict.items():
                if isinstance(value, tf.Tensor):
                    setattr(layer, attr, value.numpy().tolist())  # Convert tensor to list
        model.save(path)
    


    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
            class_weight=self.class_weights_dict  # Pass the class weights to fit
        )

        logger.info(self.model.summary())

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

