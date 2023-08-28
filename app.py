import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import load_img
from skimage.util import img_as_float
import base64
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
app = Flask(__name__)
model = pickle.load(open('braintumor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method=="POST":
        
     image = request.files['age']
     image_path = image.filename
    #  file = open('image', 'rb')
    #  byte = file.read()
    #  file.close()
    #  decodeit = open('image_path', 'wb')
    #  decodeit.write(base64.b64decode((byte)))
    #  decodeit.close()
     # img = request.get.files('age')
    #  img = request.form['age'];     #string
    #  print(image)
    #  img = Image.open(img.stream)
    #  npimg = np.fromstring(img, np.uint8)
    #  image = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
     image = image_path.astype(float)/255;
     input_arr = img_to_array(image) 
     input_arr = preprocess_input(input_arr)
     i_arr = np.array([input_arr])
    prediction = model.predict(i_arr)
    
    if prediction==0:
       return render_template('index3.html', prediction_text='NO')
    else:
       return render_template('index3.html', prediction_text='YES')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True) 