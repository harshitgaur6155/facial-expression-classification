import os
import urllib.request as request
from pathlib import Path
from zipfile import ZipFile
import tensorflow as tf
from facialExpressionClassify import logger
from facialExpressionClassify.entity.config_entity import PrepareBaseModelConfig



# ######## Uses VGG 16 Model (Old - 2014)

# class PrepareBaseModel:
#     def __init__(self, config: PrepareBaseModelConfig):
#         self.config = config


    
#     def get_base_model(self):
#         try:
#             self.model = tf.keras.applications.vgg16.VGG16(
#                 input_shape=self.config.params_image_size,
#                 weights=self.config.params_weights,
#                 include_top=self.config.params_include_top
#             )

#             self.save_model(path=self.config.base_model_path, model=self.model)

#             pass
#         except Exception as e:
#             logger.exception(e)
#             raise e


    
#     @staticmethod
#     def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
#         if freeze_all:
#             for layer in model.layers:
#                 model.trainable = False
#         elif (freeze_till is not None) and (freeze_till > 0):
#             for layer in model.layers[:-freeze_till]:
#                 model.trainable = False

#         flatten_in = tf.keras.layers.Flatten()(model.output)
#         prediction = tf.keras.layers.Dense(
#             units=classes,
#             activation="softmax"
#         )(flatten_in)

#         full_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=prediction
#         )

#         full_model.compile(
#             optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics=["accuracy"]
#         )

#         full_model.summary()
#         return full_model
    

#     def update_base_model(self):
#         self.full_model = self._prepare_full_model(
#             model=self.model,
#             classes=self.config.params_classes,
#             freeze_all=True,
#             freeze_till=None,
#             learning_rate=self.config.params_learning_rate
#         )

#         self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)





######## Uses Efficient Net B0 Model (New)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    


    def get_base_model(self):
        try:
            self.model = tf.keras.applications.EfficientNetB0(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

            self.save_model(path=self.config.base_model_path, model=self.model)
            pass

        except Exception as e:
            logger.exception(e)
            raise e
    


    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):

        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.GlobalAveragePooling2D()(model.output)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input, 
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    


    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Convert non-serializable attributes before saving the model."""
        for layer in model.layers:
            layer_dict = layer.__dict__.copy()  # Create a copy of the dictionary
            for attr, value in layer_dict.items():
                if isinstance(value, tf.Tensor):
                    setattr(layer, attr, value.numpy().tolist())  # Convert tensor to list
        model.save(path)





# ######## Uses ResNet 50 Model (New variation of VGG 16)

# class PrepareBaseModel:
#     def __init__(self, config: PrepareBaseModelConfig):
#         self.config = config
    


#     def get_base_model(self):
#         try:
#             self.model = tf.keras.applications.ResNet50(
#                 input_shape=self.config.params_image_size,
#                 weights=self.config.params_weights,
#                 include_top=self.config.params_include_top
#             )

#             self.save_model(path=self.config.base_model_path, model=self.model)
#             pass

#         except Exception as e:
#             logger.exception(e)
#             raise e
    


#     @staticmethod
#     def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):

#         if freeze_all:
#             for layer in model.layers:
#                 layer.trainable = False
#         elif freeze_till:
#             for layer in model.layers[:-freeze_till]:
#                 layer.trainable = False

#         flatten_in = tf.keras.layers.GlobalAveragePooling2D()(model.output)

#         prediction = tf.keras.layers.Dense(
#             units=classes,
#             activation="softmax"
#         )(flatten_in)

#         full_model = tf.keras.models.Model(
#             inputs=model.input, 
#             outputs=prediction
#         )

#         full_model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics=["accuracy"]
#         )

#         full_model.summary()
#         return full_model
    


#     def update_base_model(self):
#         self.full_model = self._prepare_full_model(
#             model=self.model,
#             classes=self.config.params_classes,
#             freeze_all=True,
#             freeze_till=None,
#             learning_rate=self.config.params_learning_rate
#         )

#         self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         """Convert non-serializable attributes before saving the model."""
#         for layer in model.layers:
#             layer_dict = layer.__dict__.copy()  # Create a copy of the dictionary
#             for attr, value in layer_dict.items():
#                 if isinstance(value, tf.Tensor):
#                     setattr(layer, attr, value.numpy().tolist())  # Convert tensor to list
#         model.save(path)
