import pickle

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import cm
from matplotlib import pyplot as plt
from vis.visualization import visualize_cam, overlay

from cbcNetv2 import cbcNetv2


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


def anomaly_naive_test(model):
    imgs = load_anomaly_example(anomaly=True)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        anomaly = model.predict(observation)
        print("Anomaly={}, GT=True, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))

    imgs = load_anomaly_example(anomaly=False)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        anomaly = model.predict(observation)
        print(anomaly)
        print("Anomaly={}, GT=False, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))


def model_vis(model):
    images = np.asarray(load_anomaly_example(anomaly=False))
    input_img = images[0]
    plt_img = cv2.cvtColor(input_img, cv2.COLOR_YUV2BGR)
    titles = ["Input", "Attention"]
    subplot_args = {'nrows': 1, 'ncols': 2, 'figsize': (9, 3),
                    'subplot_kw': {'xticks': [], 'yticks': []}}
    f, ax = plt.subplots(**subplot_args)
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0,
                            seed_input=input_img, grad_modifier=None)
    jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

    for i, modifier in enumerate(titles):
        ax[i].set_title(titles[i], fontsize=14)
    ax[0].imshow(plt_img)
    ax[1].imshow(overlay(plt_img, jet_heatmap, alpha=0.75))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    loaded_model = cbcNetv2.get_anomaly_inference(weigths="cbcNet-anomaly-Best_Validation.h5")
    model_vis(loaded_model)
