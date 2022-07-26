# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:32:17 2019

@author: sayadav
"""
import os, uuid, cv2, pickle
import numpy as np
import pandas as pd
from werkzeug import secure_filename
#from flask import request
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)

	# convert to RGB, if needed
	image = image.convert('RGB')

	# convert to array
	pixels = asarray(image)

	# create the detector, using default weights
	detector = MTCNN()

	# detect faces in the image
	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']

	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	# extract the face
	face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


def extract_all_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    face_list = []
    face_positions = []
        
    for facebox in results:
        # extract the bounding box from the first face
        x1, y1, width, height = facebox['box']
        
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # extract the face
        face = pixels[y1:y2, x1:x2]
        
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        face_list.append(face_array)
        face_positions.append(facebox)
        
    return face_list, face_positions
        

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
    
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
	
	# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in os.listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
    
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
    
	face_pixels = (face_pixels - mean) / std
    
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
    
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def Update_and_Save_Model_Defination(training_data_path = 'bollywood\train', facenet_model = 'facenet_keras.h5', custom_model_name = 'facenet_keras_satya'):
    trainX, trainy = load_dataset(training_data_path)
    model = load_model(facenet_model)
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
        
    trainX = asarray(newTrainX)
    
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit_transform(trainy)
    output = open(custom_model_name + '.pkl', 'wb')
    pickle.dump(out_encoder, output)
    
    trainy = out_encoder.transform(trainy)
    
    # fit model
    model = SVC(kernel='linear')
    model.fit(trainX, trainy)
    
    # save the model to disk
    filename = custom_model_name  + '.sav'
    pickle.dump(model, open(filename, 'wb'))

def put_border_and_text_on_image(img, box, label):
    x, y, width, height = box['box']
    img = cv2.rectangle(img,(x, y), (x+width, y+height), (0,255,0),2)
    #cv2.putText(img,label,(x, y-10), 2, 4,(0,255,0),2)
    return img

def get_face_positions(imagepath):
    # load image from file
    image = Image.open(imagepath)

    # convert to RGB, if needed
    image = image.convert('RGB')

    # convert to array
    pixels = np.asarray(image)

    # create the detector, using default weights
    detector = MTCNN()

    # detect faces in the image
    results = detector.detect_faces(pixels)
    return results

def get_person_name_by_encodding(facecode, datafile):
    #Check and update the encoding in csv file
    #datafile = os.path.join(APP_ROOT, "models/face_list.csv");
    if os.path.exists(datafile):
        csv_dataList = pd.read_csv(datafile)
    else:
        with open(datafile, "w") as my_empty_csv:
            csv_dataList = pd.read_csv(datafile)
    
    dataList = np.array(csv_dataList)
    
    labels = dataList[:,[0]]
    encoding = dataList[:,[np.array(range(1,dataList.shape[1]))]]
    encoding = np.squeeze(encoding, axis=1)
    
    min_distace = 100;
    min_index = -1

    for index in range(encoding.shape[0]):
        dist = np.linalg.norm(facecode - encoding[index])
        if dist < min_distace:
            min_distace = dist
            min_index = index
    
    personName = ''  
    if min_distace < 10:
        personName = labels[min_index][0]
        
    return personName

def save_uploaded_file(request, folder):
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = folder
    if not os.path.isdir(destination):
        os.mkdir(destination)
    
    filename = file.filename
    destination = os.path.join(destination, filename)
    file.save(destination)
    return filename

def save_extracted_face(folder, result_image):
    tname = str(uuid.uuid4()).replace('-','') + '.jpg'
    facename = os.path.join(folder, tname)
    cv2.imwrite(facename, result_image)
    return tname