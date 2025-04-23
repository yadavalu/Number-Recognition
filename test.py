import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


(_1, _2), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
mnist = test_images[-1000:]

# Reshape
# Last 1000 images
test_labels = test_labels[-1000:]
test_images = test_images[-1000:].reshape(-1, 28 * 28) / 255.0

model = tf.keras.models.load_model('model/my_model.keras')

y_pred = model.predict(test_images)
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print('Restored model, loss: {:5.2f}'.format(loss))

print(mnist.shape)

def model_predict(im: np.array):
    pred_arr = model.predict(im.reshape(-1, 28 * 28) / 255.0)
    max = tf.reduce_max(pred_arr)
    return tf.squeeze(tf.where(pred_arr == max))[1]


fig, ax = plt.subplots()
plt.title('Restored model, accuracy: {acc}, loss: {loss}%'.format(acc=round(100 * acc, 2), loss=round(loss, 2)))
graph = ax.imshow(np.array(mnist[0], dtype='float').reshape(28, 28), cmap='gray')
title = ax.text(0.5, 0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

total = mnist.shape[0]
correct = 0

def update(frame):
    if frame == 0:
        return

    global correct
    im = mnist[frame]
    prediction = model_predict(im)
    actual = test_labels[frame]
    
    if prediction != actual:
        fig.set_facecolor('#f8baba')
    else:
        fig.set_facecolor('#baf8ba')
        correct += 1

    title.set_text(
        "Model: {pred}, actual: {act}. Total score: {sco} ({pert}%)"
        .format(
            pred=prediction, 
            act=actual,
            sco=str(correct) + "/" + str(frame),
            pert=str(round(100 * correct / frame, 2))
        )
    )

    image = np.array(im, dtype='float')
    pixels = image.reshape((28, 28))
    graph.set_data(pixels)

anim = FuncAnimation(fig, update, frames=None)
plt.show()





