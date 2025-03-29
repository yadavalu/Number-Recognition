# Fast Number Recognition
Barebones Neutral Network for Printed and Handwritten Number Recognition trained from MNIST database

## Train and Test
To train the model even further (open weights), 
```
$ python3 train.py [options]

options:
    -e <no. of epochs>      Train for only particular number of epochs. Default is 5 epochs.
```

To test the model with the given weights,
```
$ python3 test.py
```
More functionality to be added.

## Weights and Bias cache
Weights and biases as well as number of epochs are stored in the `/cache` folder.
```
📦cache
 ┣ 📜bias_0.npy
 ┣ 📜bias_1.npy
 ┣ 📜epochs
 ┣ 📜weights_0.npy
 ┗ 📜weights_1.npy 
```
