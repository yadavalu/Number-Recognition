import glob, os, sys

import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio

import utils as u
from data.read import get_mnist

# Load the weights and biases from the trained model
weights = [
    np.load("./cache/weights_0.npy"),
    np.load("./cache/weights_1.npy"),
]

bias = [
    np.load("./cache/bias_0.npy"),
    np.load("./cache/bias_1.npy"),
]

np.set_printoptions(threshold=np.inf)

# Load the MNIST dataset
image, label = get_mnist()

# Other dataset:
for i in glob.glob("./data/2828*"):
    img_array = imageio.imread(i, mode="F") # as_gray=True (Deprication)
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img = (img_data / 255.0 * 0.99) + 0.01

    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    
    hidden = u.sigmoid(bias[0] + weights[0] @ img.reshape(784, 1))
    output = u.sigmoid(bias[1] + weights[1] @ hidden)
    print(output)

    plt.title(f"Number estimated by neural network: {output.argmax()}")
    plt.show()

while True:
    try:
        try:
            try:
                index = int(input("Enter a number (0 - 59999): "))
            except ValueError:
                print("Error: Invalid input. Please enter a number.")
                continue
            img = image[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")

            img.shape += (1,)
            # Forward propagation input -> hidden
            h_pre = bias[0] + weights[0] @ img.reshape(784, 1)
            h = 1 / (1 + np.exp(-h_pre))
            # Forward propagation hidden -> output
            o_pre = bias[1] + weights[1] @ h
            o = 1 / (1 + np.exp(-o_pre))
            print(o)

            plt.title(f"Number estimated by neural network: {o.argmax()}")
            plt.show()

        except IndexError:
            print("Error: Number exceeded 59999")
    except KeyboardInterrupt:
        print()
        exit(0)
    except EOFError:
        print()
        exit(0)
