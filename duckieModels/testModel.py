import pickle

import numpy as np

from cbcNet import cbcNet

if __name__ == "__main__":
    model = cbcNet.get_inference("cbcNet.h5")
    file = open("anomaly_img", 'rb')
    imgs = pickle.load(file)
    file.close()
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        prediction, anomaly = model.predict(observation)
        print(prediction[0][0], prediction[0][1])
        print(anomaly[0][0] < 0.5)
