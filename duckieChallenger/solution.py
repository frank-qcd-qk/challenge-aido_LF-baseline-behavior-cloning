#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
from aido_schemas import (Context, DB20Commands, DB20Observations, EpisodeStart, JPGImage, LEDSCommands,
                          logger, protocol_agent_DB20, PWMCommands, RGB, wrap_direct)
from frankModel import FrankNet
from helperFncs import SteeringToWheelVelWrapper, image_resize

#! Global Config
expect_shape = (480, 640, 3)
convertion_wrapper = SteeringToWheelVelWrapper()


class TensorflowTemplateAgent:

    def __init__(self):
        pass

    def init(self, context: Context):
        context.info("Check GPU...")
        self.check_tensorflow_gpu()
        context.info('init()')
        self.model = FrankNet.build(200, 150)
        self.model.load_weights("FrankNet.h5")
        self.current_image = np.zeros(expect_shape)
        self.input_image = np.zeros((150, 200, 3))
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

    def check_tensorflow_gpu(self):
        req = os.environ.get('AIDO_REQUIRE_GPU', None)
        name = tf.test.gpu_device_name()
        logger.info(f'gpu_device_name: {name!r} AIDO_REQUIRE_GPU = {req!r}')
        if req is not None:
            if not name:  # None or ''
                msg = 'Could not find gpu device.'
                logger.error(msg)
                raise RuntimeError(msg)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    #! Image pre-processing here
    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)
        self.input_image = image_resize(self.current_image, width=200)
        self.input_image = self.input_image[0:150, 0:200]
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2YUV)
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

    #! Modification here! Return with action
    def compute_action(self, observation):
        prediction = self.model.predict(observation)
        print(prediction)
        print(prediction.shape)
        return prediction[0], prediction[1]

    #! Major Manipulation here Should not always change
    def on_received_get_commands(self, context: Context):
        linear, angular = self.compute_action(self.to_predictor)
        #! Inverse Kinematics
        pwm_left, pwm_right = convertion_wrapper.convert(linear, angular)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        #! LED Commands Sherrif Duck
        grey = RGB(0.0, 0.0, 0.0)
        red = RGB(255.0, 0.0, 0.0)
        blue = RGB(0.0, 0.0, 255.0)
        led_commands = LEDSCommands(red, grey, blue, red, blue)

        #! Send PWM Command
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data





def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
