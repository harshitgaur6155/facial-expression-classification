import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
import time
from facialExpressionClassify.entity.config_entity import PrepareCallbacksConfig



class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
    


    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    


    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )
    


    @property
    def _create_lr_scheduler_callback(self):
        # Implement ReduceLROnPlateau callback
        return ReduceLROnPlateau(
            monitor='val_loss',   # Or 'val_accuracy'
            factor=0.1,           # Reduce LR by a factor of 0.1
            patience=3,           # Number of epochs to wait before reducing LR
            min_lr=1e-7           # Minimum learning rate
        )
    

    
    def get_tb_ckpt_lr_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            # self._create_lr_scheduler_callback  # Include the LR scheduler callback
        ]
