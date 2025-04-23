# Tensorflow Number Recognition
Neutral Network using Tensorflow for Printed and Handwritten Number Recognition trained from MNIST database

Checkout branch `bare-bones` to view the same project built from scratch without Tensorflow.
Check [bare-bones](https://github.com/yadavalu/Number-Recognition/tree/bare-bones)

## Train and Test
To train the model, run
```
$ python3 train.py
```

To test the model with the given weights,
```
$ python3 test.py
```

## Other funcitonalities
test.py: flashes through the different images.

More functionality to be added.

## Weights and Bias cache
Weights and biases as well as number of epochs are stored in the `/model` folder.
```
📦model
 ┣ 📜my_model.keras
 ┣ 📜.gitignore
```
