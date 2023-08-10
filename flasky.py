# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:06:37 2023

@author: batma
"""

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

app = Flask(__name__,static_folder='static')
file_path = os.path.join(os.getcwd(), 'mushroom.h5')
model = load_model(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/mushrooms.html')
def mushrooms():
    return render_template('mushrooms.html')

@app.route('/prediction.html', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)

        index = ['Boletus', 'Lactarius', 'Russula']

        result = index[prediction[0]]

        # Delete the uploaded image
        os.remove(filepath)

        return render_template('prediction.html', result=result)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
