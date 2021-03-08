import pickle

import numpy as np
import tensorflow as tf
from matplotlib import cm
from matplotlib import pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

from cbcNet import cbcNet


def get_model_summary(model):
    print(model.summary())
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=True, dpi=128
    )


def load_anomaly_example(anomaly):
    if anomaly:
        file = open("anomaly_img", 'rb')
    else:
        file = open("normal_img", 'rb')
    imgs = pickle.load(file)
    file.close()
    return imgs


def single_test(model):
    output = model.predict(np.asarray(load_anomaly_example(anomaly=True)))
    print(output[1][1])


def naive_test(model):
    imgs = load_anomaly_example(anomaly=True)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        prediction, anomaly = model.predict(observation)
        print("Anomaly={}, GT=True, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))
        print(prediction[0][0], prediction[0][1])

    imgs = load_anomaly_example(anomaly=False)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        prediction, anomaly = model.predict(observation)
        print("Anomaly={}, GT=False, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))
        print(prediction[0][0], prediction[0][1])


def model_vis(model):
    images = np.asarray(load_anomaly_example(anomaly=True))

    # Rendering
    image_titles = ['Anomaly1', 'Anomaly2', 'Anomaly3', 'Anomaly4', 'Anomaly5']
    subplot_args = {'nrows': 1, 'ncols': 5, 'figsize': (9, 3),
                    'subplot_kw': {'xticks': [], 'yticks': []}}
    f, ax = plt.subplots(**subplot_args)
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(images[i])
    plt.tight_layout()
    plt.show()

    # Then, when the softmax activation function is applied to the last layer of model,
    # it may obstruct generating the attention images, so you need to replace the function
    # to a linear function. Here, we does so using model_modifier.
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m

    # TODO: Score_function breaks things...
    def score_function(output):
        return (tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(1))

    # Create Gradcam object
    gradcam = Gradcam(model,
                      model_modifier=model_modifier,
                      clone=False)

    # Generate heatmap with GradCAM
    cam = gradcam(score_function,
                  images,
                  penultimate_layer=-1,  # model.layers number
                  )
    cam = normalize(cam)
    f, ax = plt.subplots(**subplot_args)
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)  # overlay
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    loaded_model = cbcNet.get_inference("cbcNet_CallbackMethod.h5")
    model_vis(loaded_model)
