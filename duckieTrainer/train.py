import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from duckieLog.log_util import Reader
from duckieModels.cbcNet import cbcNet

MODEL_NAME = "cbcNet"
logging.basicConfig(level=logging.INFO)

# ! Default Configuration
EPOCHS = 10
INIT_LR = 1e-3
BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
TRAINING_DATASET = "train.log"


class DuckieTrainer:
    def __init__(
            self,
            epochs,
            init_lr,
            batch_size,
            log_file,
            split,
    ):

        # 0. Setup Folder Structure
        self.create_dirs()
        exit()

        # 1. Load Data
        try:
            self.observation, self.linear, self.angular = self.get_data(
                log_file
            )
        except Exception:
            logging.error("Loading dataset failed... exiting...")
            exit(1)
        logging.info(f"Loading Datafile completed")

        # 2. Split training and testing
        (
            observation_train,
            observation_valid,
            linear_train,
            linear_valid,
            angular_train,
            angular_valid,
            anomaly_train,
            anomaly_valid

        ) = train_test_split(
            self.observation, self.linear, self.angular, self.anomaly, test_size=1 - split, shuffle=True
        )

        prediction_train = np.array(list(zip(linear_train, angular_train)))
        prediction_valid = np.array(list(zip(linear_valid, angular_valid)))

        model = cbcNet.get_model(init_lr, epochs)

        callbacks_list = self.configure_callbacks()

        # 11. GO!
        history = model.fit(
            x=observation_train,
            y=prediction_train,
            validation_data=(
                observation_valid,
                prediction_valid,
            ),
            epochs=EPOCHS,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        model.save(f"trainedModel/{MODEL_NAME}.h5")

    def create_dirs(self):
        try:
            dirname, _ = os.path.split(os.path.abspath(__file__))
            Path(os.path.join(dirname, "trainedModel")).mkdir(parents=True, exist_ok=True)
        except OSError:
            print(
                "Create folder for trained model failed. Please check system permissions."
            )
            exit()

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(
                f'{MODEL_NAME}-{datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}'
            )
        )

        filepath1 = f"trainedModel/{MODEL_NAME}Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}Best_Loss.h5"
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
            filepath2, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )

        return [checkpoint1, checkpoint2, tensorboard]

    def get_data(self, file_path, old_dataset=False):
        """
        Returns (observation: np.array, linear: np.array, angular: np.array)
        """
        reader = Reader(file_path)

        observation, linear, angular = (
            reader.modern_read()
        )

        logging.info(
            f"""Observation Length: {len(observation)}
            Linear Length: {len(linear)}
            Angular Length: {len(angular)}"""
        )
        return np.array(observation), np.array(linear), np.array(angular)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")

    parser.add_argument(
        "--epochs", help="Set the total training epochs", default=EPOCHS
    )
    parser.add_argument(
        "--learning_rate", help="Set the initial learning rate", default=INIT_LR
    )
    parser.add_argument(
        "--batch_size", help="Set the batch size", default=BATCH_SIZE)
    parser.add_argument(
        "--training_dataset", help="Set the training log file name", default=TRAINING_DATASET
    )
    parser.add_argument(
        "--split", help="Set the training and test split point (input the percentage of training)",
        default=TRAIN_PERCENT
    )

    args = parser.parse_args()

    DuckieTrainer(
        epochs=int(args.epochs),
        init_lr=float(args.learning_rate),
        batch_size=int(args.batch_size),
        log_file=args.training_dataset,
        split=float(args.split)
    )
