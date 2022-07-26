from flask import Flask, request, render_template
from werkzeug import secure_filename
import numpy as np
import pandas as pd
import re, base64, io, uuid, os, cv2
from keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from keras import backend as K
from flask import jsonify
import FaceNet_satya_util as satya
import csv
import nlp_util as nlp_satya
from keras.preprocessing.text import Tokenizer
import nlp_prediction_functios as nlp_predict

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.realpath(__file__))
#APP_ROOT = os.path.dirname(os.path.curdir)

#facenet_keras = load_model(os.path.join(APP_ROOT, "models", "facenet_keras.h5"))

   
def factors(num):
    return [x for x in range(1, num+1) if num%x==0]

@app.route('/')
def hello_world():
    return render_template(
            "./home.html"  # name of template
            )
    
@app.route('/image_prediction')
def image_prediction():
    return render_template(
            "./image_prediction.html"  # name of template
            )

@app.route('/digit_recognizer')
def digit_recognizer():
    return render_template(
            "./digit_recognizer.html"  # name of template
            )

@app.route('/cloth_recognizer')
def cloth_recognizer():
    return render_template(
            "./cloth_recognizer.html"  # name of template
            )
@app.route('/face_extractor')
def face_extractor():
    return render_template(
            "./face_extractor.html"  # name of template
            )
@app.route('/face_recognizer')
def face_recognizer():
    return render_template(
            "./face_recognizer.html"  # name of template
            )
    
@app.route('/nlp_sentiment_movie_review')
def nlp_sentiment_movie_review():
    return render_template(
            "./nlp_sentiment_movie_review.html"  # name of template
            )
@app.route('/nlp_news_classification_01')
def nlp_news_classification_01():
    return render_template(
            "./nlp_news_classification_01.html"  # name of template
            )

@app.route('/nlp_next_word_prediction_01')
def nlp_next_word_prediction_01():
    return render_template(
            "./nlp_next_word_prediction_01.html"  # name of template
            )
    
@app.route('/nlp_english_to_french')
def nlp_english_to_french():
    return render_template(
            "./nlp_english_to_french.html"  # name of template
            )
    
@app.route('/predict_image', methods=['GET','POST'])
def predict_image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = os.path.join(APP_ROOT,"images")
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    filename = file.filename
    destination = os.path.join(destination, filename)
    file.save(destination)
    
    #Make the prediction
    classifier = load_model(os.path.join(APP_ROOT, "models", "cnn_satya_dataset_64X64_10000.h5"))
    test_image = image.load_img(destination, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    K.clear_session()
    prediction = 'dog'
    
    if(result[0][0] == 0):
        prediction = 'cat'

    # logic to load image
    return prediction, 200


@app.route('/identify_digit', methods=['GET','POST'])
def identify_digit():
    imgstring=request.form['file']
    base64_data = re.sub('data:image/.+;base64,', '', imgstring)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    
    destination = os.path.join(APP_ROOT,"images")
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    input_folder = os.path.join(destination,'test.png')
    print(input_folder)
    img.save(input_folder, "png") 
    
    #Make prediction
    model = load_model(os.path.join(APP_ROOT, 'models', 'digit_recognizer.h5'))

    test_image = image.load_img(input_folder, target_size=(28,28),color_mode = "grayscale",grayscale=True)
    test_image = np.array(test_image)
    
    num_pixels = test_image.shape[0] * test_image.shape[1] #it will be 784
    
    # change the imege array of type (60000, 28,28) to (60000,784)
    test_image = test_image.reshape(1,num_pixels).astype('float32')
    test_image = test_image / 255
    
    result = model.predict(test_image)
    K.clear_session()
    #a = {'result':result, 'prediction': np.argmax(a)}
    #np.argmax(a)
    # logic to load image
    return str(np.argmax(result)), 200

@app.route('/predict_cloth', methods=['GET','POST'])
def predict_cloth():
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = os.path.join(APP_ROOT,"images")
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    filename = file.filename
    destination = os.path.join(destination, filename)
    file.save(destination)
    
    #Make the prediction
    classifier = load_model(os.path.join(APP_ROOT, "models", "cloth_model.h5"))
    test_image = image.load_img(destination, target_size=(28,28), grayscale=True)
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    K.clear_session()
    prediction = np.argmax(result)
    cloth = pd.read_csv(os.path.join(APP_ROOT, "models", "cloth_types.csv"))
    prediction = cloth.iloc[prediction][1]

    # logic to load image
    return prediction, 200

@app.route('/extract_faces', methods=['GET','POST'])
def extract_faces():
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = os.path.join(APP_ROOT,"images")
    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    filename = file.filename
    destination = os.path.join(destination, filename)
    file.save(destination)
    
    # Extract the faces
    imagePath = destination
    cascPath = os.path.join(APP_ROOT,"static/cascades/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # SEARCH CORDINATE OF FACES
    faces = face_cascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5
    )
    facelist = []
    facepath = os.path.join(APP_ROOT,'static/faces')
    if not os.path.isdir(facepath):
        os.mkdir(facepath)
    
    for x,y,w,h in faces:
        roi_color = image[y:y + h, x:x + w]
        tname = str(uuid.uuid4()).replace('-','') + '.jpg'
        facename = os.path.join(facepath, tname)
        cv2.imwrite(facename, roi_color)
        facelist.append(tname)
    
    # mark the border on identified faces
    for x,y,w,h in faces:
        image = cv2.rectangle(image,(x, y), (x+w, y+h), (0,255,0),2)
    
    # make the prediction on extracted faces
    
    tname = str(uuid.uuid4()).replace('-','') + '.jpg'
    facename = os.path.join(facepath, tname)
    cv2.imwrite(facename, image)
    facelist.append(tname)
            
    return jsonify(facelist), 200

@app.route('/recognize_single_faces', methods=['GET','POST'])
def recognize_single_faces():
    destination = os.path.join(APP_ROOT,"images")
    filename = satya.save_uploaded_file(request, destination)
    destination = os.path.join(destination, filename)
    
    #load pretrained model
    model = load_model(os.path.join(APP_ROOT, "models", "facenet_keras.h5"))
    
    all_faces, face_positions = satya.extract_all_face(destination)
    datafile = os.path.join(APP_ROOT, "models/face_list.csv");
    facepath = os.path.join(APP_ROOT,'static/faces')
    facelist = []
    
    for index, face_pixel in enumerate(all_faces):
        face_encodeing = satya.get_embedding(model, face_pixel)
        
        #Check the encoding in csv file
        predictedName = satya.get_person_name_by_encodding(face_encodeing, datafile)
        if len(predictedName) == 0:
            predictedName = "unknown"
        
        tname = satya.save_extracted_face(facepath, face_pixel)
        dict1 = { 'Name' : predictedName, 'Image' : tname } 
        facelist.append(dict1)
    
    K.clear_session()
    
    #create borders on main image to shocase the face identification
    result_image = cv2.imread(destination)
    for fpos in face_positions:
        result_image = satya.put_border_and_text_on_image(result_image, fpos, 'NA')
    
    tname = satya.save_extracted_face(facepath, result_image)
    dict1 = { 'Name' : predictedName, 'Image' : tname } 
    facelist.append(dict1)
    
    return jsonify(facelist), 200

@app.route('/upload_new_faces/<name>', methods=['GET','POST'])
def upload_new_faces(name):
    destination = os.path.join(APP_ROOT,"images")
    filename = satya.save_uploaded_file(request, destination)
    personName = os.path.splitext(filename)[0]
    destination = os.path.join(destination, filename)
    
    face_pixel = satya.extract_face(destination)

    #load pretrained model
    model = load_model(os.path.join(APP_ROOT, "models", "facenet_keras.h5"))
    
    face_encodeing = satya.get_embedding(model, face_pixel)
    K.clear_session()
    
    #Check and update the encoding in csv file
    datafile = os.path.join(APP_ROOT, "models/face_list.csv");
    predictedName = satya.get_person_name_by_encodding(face_encodeing, datafile)
    
    if len(predictedName) > 0:
        personName = predictedName
    else:
        add_person_name_with_encodding(name, face_encodeing, datafile)

    return personName, 200
    

def add_person_name_with_encodding(personName, newface, datafile):
    newface = np.append(personName, newface)
    newface = newface.reshape(1,129)
    with open(datafile, 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(newface.tolist())
        writeFile.close()

############## NLP CODE START HERE ##########################
@app.route('/nlp_check_sentiment_movie_review/<text>', methods=['GET','POST'])
def nlp_check_sentiment_movie_review(text):
    text = request.form['inputText']
    
    import urllib.parse
    text = urllib.parse.unquote_plus(text)
    reviews_test = []
    reviews_test.append(text)
    reviews_test = nlp_satya.CleanAndRemoveHtmlAndOtherSpacialChars(reviews_test)
    reviews_test = nlp_satya.NTLK_remove_stop_words(reviews_test)
    reviews_test = nlp_satya.NTLK_get_stemmed_text(reviews_test)
    reviews_test = nlp_satya.NTLK_get_lemmatized_text(reviews_test)
    
    #load pretrained model
    model = nlp_satya.pickle_LoadModal(os.path.join(APP_ROOT, "models", "Movie_Review.h5"))
    ngram_vectorizer = nlp_satya.pickle_LoadModal(os.path.join(APP_ROOT, "models", "Movie_Review_ngram_vectorizer.h5"))
    reviews_test = ngram_vectorizer.transform(reviews_test)
    prediction = model.predict(reviews_test)
    K.clear_session()
    return str(prediction[0]), 200

@app.route('/nlp_predict_news_classification_01', methods=['GET','POST'])
def nlp_predict_news_classification_01():
    text = request.form['inputText']
    
    import urllib.parse
    text = urllib.parse.unquote_plus(text)
    model_input = []
    model_input.append(text)
        
    #load pretrained model
    model = load_model(os.path.join(APP_ROOT, "models", "model_news_classification_01.h5"))
    tokenizer  = Tokenizer()
    tokenizer = nlp_satya.pickle_LoadModal(os.path.join(APP_ROOT,"models","tokenizer_news_classification_01.pickle"))
    model_input = tokenizer.texts_to_matrix(model_input, mode='tfidf')
    prediction = model.predict(model_input)
    K.clear_session()
    return str(np.argmax(prediction)), 200
#    return 'ok',200
    
@app.route('/nlp_predict_next_word_01', methods=['GET','POST'])
def nlp_predict_next_word_01():
    text = request.form['inputText']
    result = nlp_predict.nlp_predict_next_word_01(APP_ROOT, text)
    return result, 200

@app.route('/nlp_convert_english_to_french', methods=['GET','POST'])
def nlp_convert_english_to_french():
    text = request.form['inputText']
    modelName = request.form['modelName']
    import nlp_english_to_french as convertor
    result = ''
    if (modelName == 'eng-fra'):
        result = convertor.convert_english_to_french(APP_ROOT, text)
    else:
        result = convertor.convert_english_to_hindi(APP_ROOT, text)
    return result, 200
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)