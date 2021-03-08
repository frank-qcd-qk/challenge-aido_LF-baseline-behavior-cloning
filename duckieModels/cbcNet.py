import tensorflow as tf
from tensorflow.keras.backend import less_equal as Less_equal
from tensorflow.keras.backend import switch as Switch
from tensorflow.keras.layers import Conv2D, Lambda, Flatten, Dense


class cbcNet:
    @staticmethod
    def build_cbc_net(rgb_image):
        # ? Input Normalization
        normalized_image = Lambda(lambda x: x / 255.0)(rgb_image)

        # ? Behavior Cloning:
        # ? L1: CONV => RELU
        bc_branch = Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation='relu', name='BC_Conv1')(
            normalized_image)
        # ? L2: CONV => RELU
        bc_branch = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='BC_Conv2')(bc_branch)
        # ? L3: CONV => RELU
        bc_branch = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='BC_Conv3')(bc_branch)
        # ? L4: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='BC_Conv4')(bc_branch)
        # ? L5: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='BC_Conv5')(bc_branch)
        # ? Flatten
        bc_branch = Flatten()(bc_branch)

        # ? Anomaly Detector:
        # ? L1: CONV => RELU
        anomaly_branch = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='AN_Conv1')(
            normalized_image)
        # ? L2: CONV => RELU
        anomaly_branch = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='AN_Conv2')(
            anomaly_branch)
        # ? L3: CONV => RELU
        anomaly_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='AN_Conv3')(anomaly_branch)
        # ? Flatten
        anomaly_branch = Flatten()(anomaly_branch)
        # ? Fully Connected
        anomaly_branch = Dense(50, kernel_initializer='normal', activation='relu', name='AN_FC1')(anomaly_branch)
        anomaly_branch = Dense(10, kernel_initializer='normal', activation='relu', name='AN_FC2')(anomaly_branch)
        anomaly = Dense(1, kernel_initializer='normal', activation='sigmoid', name="Anomaly_Out")(anomaly_branch)

        # ? Initial Fully Connected
        prediction = Dense(1164, kernel_initializer='normal', activation='relu', name='BC_FC1')(bc_branch)
        prediction = Dense(500, kernel_initializer='normal', activation='relu', name='BC_FC2')(prediction)

        x = Dense(500, kernel_initializer='normal', activation='relu', name='ANB_FC1')(prediction)
        x = Dense(50, kernel_initializer='normal', activation='relu', name='ANB_FC2')(x)
        x = Dense(10, kernel_initializer='normal', activation='relu', name='ANB_FC3')(x)
        x = Dense(2, kernel_initializer='normal', name='ANB_Out')(x)

        y = Dense(500, kernel_initializer='normal', activation='relu', name='BCB_FC1')(prediction)
        y = Dense(50, kernel_initializer='normal', activation='relu', name='BCB_FC2')(y)
        y = Dense(10, kernel_initializer='normal', activation='relu', name='BCB_FC3')(y)
        y = Dense(2, kernel_initializer='normal', name='BCB_Out')(y)

        # ? Switch
        prediction = Switch(Less_equal(anomaly, 0.5), x, y, name="prediction")
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
        losses = {"tf.keras.backend.switch": "mse", "Anomaly_Out": "mse"}
        loss_weights = {"tf.keras.backend.switch": 10, "Anomaly_Out": 1}

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


if __name__ == "__main__":
    model = cbcNet.get_model(lr=0.1, epochs=1000)
    print(model.summary())
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=True, dpi=128
    )
