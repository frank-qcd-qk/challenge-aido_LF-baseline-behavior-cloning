import argparse
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from frankModel import FrankNet
from sklearn.model_selection import train_test_split

from log_reader import Reader

MODEL_NAME = "FrankNet"
logging.basicConfig(level=logging.INFO)

# ! Default Configuration
EPOCHS = 10
INIT_LR = 1e-3
BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
LOG_DIR = "/home/duckie/challenge-aido_LF-baseline-behavior-cloning/duckieTrainer/"
LOG_FILE = "train.log"

EXPERIMENTAL = False
OLD_DATASET = False


class DuckieTrainer:
    def __init__(
            self,
            epochs,
            init_lr,
            batch_size,
            log_dir,
            log_file,
            old_dataset,
            experimental,
            split,
    ):
        print("Observed TF Version: ", tf.__version__)
        print("Observed Numpy Version: ", np.__version__)

        self.create_dir()

        # 1. Load all the datas
        log_path = os.path.join(log_dir, log_file)
        logging.info(f"Loading Datafile {log_path}")
        try:
            self.observation, self.linear, self.angular = self.get_data(
                log_file, old_dataset
            )
        except Exception:
            try:
                self.observation, self.linear, self.angular = self.get_data(
                    log_file, old_dataset
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
        ) = train_test_split(
            self.observation, self.linear, self.angular, test_size=1 - split, shuffle=True
        )

        prediction_train = np.array(list(zip(linear_train, angular_train)))
        prediction_valid = np.array(list(zip(linear_valid, angular_valid)))

        model = self.configure_model(lr=init_lr, epochs=epochs)

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

    def create_dir(self):
        try:
            os.makedirs("trainedModel")
        except FileExistsError:
            print("Directory already exists!")
        except OSError:
            print(
                "Create folder for trained model failed. Please check system permissions."
            )
            exit()

    def configure_model(self, lr, epochs):
        model = FrankNet.build(200, 150)
        opt = tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs)
        model.compile(
            optimizer=opt, loss="mse", metrics="mse"
        )
        return model

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
            reader.read() if old_dataset else reader.modern_read()
        )

        logging.info(
            f"""Observation Length: {len(observation)}
            Linear Length: {len(linear)}
            Angular Length: {len(angular)}"""
        )
        return (np.array(observation), np.array(linear), np.array(angular))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")

    parser.add_argument(
        "--experimental",
        help="Set if to use the experimental data loading method.",
        action="store_true",
        default=EXPERIMENTAL,
    )
    parser.add_argument(
        "--old_dataset",
        help="Set to use the old data log format",
        action="store_true",
        default=OLD_DATASET,
    )
    parser.add_argument(
        "--epochs", help="Set the total training epochs", default=EPOCHS
    )
    parser.add_argument(
        "--learning_rate", help="Set the initial learning rate", default=INIT_LR
    )
    parser.add_argument(
        "--batch_size", help="Set the batch size", default=BATCH_SIZE)
    parser.add_argument(
        "--log_dir", help="Set the training log directory", default=LOG_DIR
    )
    parser.add_argument(
        "--log_file", help="Set the training log file name", default=LOG_FILE
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
        log_dir=args.log_dir,
        log_file=args.log_file,
        old_dataset=args.old_dataset,
        experimental=args.experimental,
        split=float(args.split)
    )
