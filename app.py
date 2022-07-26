from flask import Flask, request, render_template
from werkzeug import secure_filename
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template(
            "./home.html"  # name of template
            )

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=80)
