import tensorflow as tf
import numpy as np
import gradio as gr

model = tf.keras.models.load_model("Digit.h5")


def predict_digit(img):
    img=img.reshape((-1,28,28))
    img=img/255
    return np.argmax(model.predict(img))


iface = gr.Interface(predict_digit, inputs="sketchpad", outputs="label")
iface.launch(inbrowser=True,debug='true')