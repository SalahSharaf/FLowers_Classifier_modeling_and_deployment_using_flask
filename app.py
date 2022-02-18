from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np


app=Flask(__name__)

app.config['UPLOAD_FOLDER']='uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMAGE_RES=224
model=tf.keras.models.load_model('my_model')
classes=['dandelion' ,'daisy' ,'tulips' ,'sunflowers' ,'roses']




def predict_func(image_path):
    image=tf.keras.utils.load_img(image_path)   
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    print(image.shape)
    prediction=model.predict(image[None ,...])
    print(prediction[0])
    prediction_index=np.argmax(prediction)
    print(prediction_index)
    result=classes[prediction_index]
    probability=prediction[0][prediction_index]
    print(result,'   ',probability)
    return result,probability

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


print(allowed_file('sss.jpg'))
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      if(allowed_file(f.filename)):
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          result,probability=predict_func(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          return render_template('result.html',result=result,probability=probability)
      else:
          return render_template('result.html')

  
if __name__=='__main__':
    app.run(host='127.0.0.2',port=4000)


