import tensorflow as tf
import time
import os
from cnnClassifier.entity import PrepareCallbacksConfig
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class PrepareCallback:
    def __init__(self, config):
        self.config = config

    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    def _create_ckpt_callbacks(self):
        checkpoint_dir = os.path.join(
            self.config.checkpoint_root_dir,
            "model_checkpoint",
        )
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            # other parameters...
        )

    def get_tb_ckpt_callbacks(self):
        checkpoint_callback = ModelCheckpoint(filepath="path_to_save_checkpoints", save_best_only=True)
        early_stopping_callback = EarlyStopping(patience=5, monitor='val_loss')
        tensorboard_callback = TensorBoard(log_dir="tensorboard_log_directory")

        return [checkpoint_callback, early_stopping_callback, tensorboard_callback]
    

    