import glob, os, sys

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import imageio.v2 as imageio

import utils as u
from data.read import get_mnist


print("Loading MNIST DB ...")
image, label = get_mnist()


weights = [
    np.random.uniform(-0.5, 0.5, (20, 784)),
    np.random.uniform(-0.5, 0.5, (10, 20)),
]

bias = [
    np.zeros((20, 1)),
    np.zeros((10, 1)),
]

if os.path.isfile("./cache/weights_0.npy"):
    weights[0] = np.load("./cache/weights_0.npy")
    print("Restoring cached weights_0 ...")
if os.path.isfile("./cache/weights_1.npy"):
    weights[1] = np.load("./cache/weights_1.npy")
    print("Restoring cached weights_1 ...")
if os.path.isfile("./cache/bias_0.npy"):
    bias[0] = np.load("./cache/bias_0.npy")
    print("Restoring cached bias_0 ...")
if os.path.isfile("./cache/bias_1.npy"):
    bias[1] = np.load("./cache/bias_1.npy")
    print("Restoring cached bias_1 ...")

epoch = 0
if os.path.isfile("./cache/epochs"):
    f = open("./cache/epochs", "r")
    epoch = int(f.read())
    f.close()

print("Starting neural learning ...")

r = 0.01
epochs = 5
if sys.argv[1] == "-e":
    epochs = int(sys.argv[2])
    print("Running for " + str(epochs) + " epochs ...\n")

for i in range(epochs):
    highest = 0
    print("\nEpoch: " + str(i + 1 + epoch))
    for img, lab in zip(image, label):
        # Forward propagation
        img.shape += (1,)
        lab.shape += (1,)

        hidden_neuron = u.sigmoid(weights[0] @ img + bias[0])
        output_neuron = u.sigmoid(weights[1] @ hidden_neuron + bias[1])

        error = u.mean_error_function(output_neuron, lab)
        highest += int(np.argmax(output_neuron) == np.argmax(lab))

        # Backpropagation:
        do = output_neuron - lab
        weights[1] += -r * do @ np.transpose(hidden_neuron)
        bias[1] += -r * do

        dh = np.transpose(weights[1]) @ do * u.sigmoid_dash(hidden_neuron)
        weights[0] += -r * dh @ np.transpose(img)
        bias[0] += -r * dh

    print(f"Accuracy: {round((highest / image.shape[0]) * 100, 2)}%")
    print(str(highest) + "/" + str(60000))

print("Writing to cache ...")

np.save("./cache/weights_0.npy", weights[0])
np.save("./cache/weights_1.npy", weights[1])
np.save("./cache/bias_0.npy", bias[0])
np.save("./cache/bias_1.npy", bias[1])

f = open("./cache/epochs", "w")
f.write(str(epochs + epoch))
f.close()

