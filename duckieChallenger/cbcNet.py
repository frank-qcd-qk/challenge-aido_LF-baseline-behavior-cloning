import tensorflow as tf
from tensorflow.keras.backend import less_equal as Less_equal
from tensorflow.keras.backend import switch as Switch
from tensorflow.keras.layers import Conv2D, Activation, Lambda, Flatten, Dense


class cbcNet:
    @staticmethod
    def build_cbc_net(rgb_image):
        # ? Input Normalization
        normalized_image = Lambda(lambda x: x / 255.0)(rgb_image)

        # ? Behavior Cloning:
        # ? L1: CONV => RELU
        bc_branch = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(normalized_image)
        bc_branch = Activation("relu")(bc_branch)
        # ? L2: CONV => RELU
        bc_branch = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(bc_branch)
        bc_branch = Activation("relu")(bc_branch)
        # ? L3: CONV => RELU
        bc_branch = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(bc_branch)
        bc_branch = Activation("relu")(bc_branch)
        # ? L4: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), padding="valid")(bc_branch)
        bc_branch = Activation("relu")(bc_branch)
        # ? L5: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), padding="valid")(bc_branch)
        bc_branch = Activation("relu")(bc_branch)
        # ? Flatten
        bc_branch = Flatten()(bc_branch)

        # ? Anomaly Detector:
        # ? L1: CONV => RELU
        anomaly_branch = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(normalized_image)
        anomaly_branch = Activation("relu")(anomaly_branch)
        # ? L2: CONV => RELU
        anomaly_branch = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(anomaly_branch)
        anomaly_branch = Activation("relu")(anomaly_branch)
        # ? L3: CONV => RELU
        anomaly_branch = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(anomaly_branch)
        anomaly_branch = Activation("relu")(anomaly_branch)
        # ? Flatten
        anomaly_branch = Flatten()(anomaly_branch)
        # ? Fully Connected
        anomaly_branch = Dense(50, kernel_initializer='normal', activation='relu')(anomaly_branch)
        anomaly_branch = Dense(10, kernel_initializer='normal', activation='relu')(anomaly_branch)
        anomaly = Dense(1, kernel_initializer='normal', activation='sigmoid', name="Anomaly")(anomaly_branch)

        # ? Initial Fully Connected
        prediction = Dense(1164, kernel_initializer='normal', activation='relu')(bc_branch)
        prediction = Dense(100, kernel_initializer='normal', activation='relu')(prediction)

        def anomaly_controller(x):
            x = Dense(50, kernel_initializer='normal', activation='relu')(x)
            x = Dense(10, kernel_initializer='normal', activation='relu')(x)
            x = Dense(2, kernel_initializer='normal')(x)
            return x

        def lane_following_controller(y):
            y = Dense(50, kernel_initializer='normal', activation='relu')(y)
            y = Dense(10, kernel_initializer='normal', activation='relu')(y)
            y = Dense(2, kernel_initializer='normal')(y)
            return y

        # ? Switch
        prediction = Switch(Less_equal(anomaly, 0.5), anomaly_controller(prediction),
                            lane_following_controller(prediction), name="Command")
        return prediction, anomaly

    @staticmethod
    def get_model(lr, epochs, input_shape=(150, 200, 3)):
        # ! Define input
        rgb_input = tf.keras.Input(shape=input_shape)
        # TODO: Add Velocity Input

        # ! Build Structure
        (driving_cmd, anomaly_detection) = cbcNet.build_cbc_net(rgb_input)
        model = tf.keras.Model(inputs=rgb_input, outputs=[driving_cmd, anomaly_detection], name="cbcNet")
        # ! Setup Optimizer
        opt = tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs)
        # ! Compile Model
        losses = {"tf.keras.backend.switch": "mse", "Anomaly": "mse"}
        loss_weights = {"tf.keras.backend.switch": 1, "Anomaly": 5}

        model.compile(
            optimizer=opt, loss=losses, loss_weights=loss_weights
        )
        return model

    @staticmethod
    def get_inference(weigths="cbcNet.h5", input_shape=(150, 200, 3)):
        rgb_input = tf.keras.Input(shape=input_shape)
        (driving_cmd, anomaly_detection) = cbcNet.build_cbc_net(rgb_input)
        model = tf.keras.Model(inputs=rgb_input, outputs=[driving_cmd, anomaly_detection], name="cbcNet")
        model.load_weights(weigths)
        return model
