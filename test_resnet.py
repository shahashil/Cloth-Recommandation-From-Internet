import tensorflow 
import keras
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
import operations as op
import pickle
import warnings
from sklearn.neighbors import NearestNeighbors
import cv2

warnings.filterwarnings('ignore')

def get_model():
    model = ResNet50(weights = 'imagenet' , include_top = False , input_shape = (224,224,3))
    model.trainable = False
    model = Sequential([model,GlobalMaxPooling2D()])  #tensorflow.keras.
    return model


def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preproccessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preproccessed_img).flatten()
    normalized_result = result/ norm(result)
    return normalized_result

def test_similarity(uid,img_name):
    filenames_str = str(uid)+'-filenames.pkl'
    features_str = str(uid)+'-features.pkl'
    
    op.download_bucket_obj('pkl',filenames_str)
    op.download_bucket_obj('pkl',features_str)
    
    old_filenames = pickle.load(open(filenames_str,'rb'))
    old_features = pickle.load(open(features_str,'rb'))

    op.download_bucket_obj('wardrobe',img_name)

    neighbors = NearestNeighbors(n_neighbors = 2,algorithm = 'brute',metric = 'euclidean')
    neighbors.fit(old_features)

    test_img = extract_features(img_name,get_model())

    distances,indices = neighbors.kneighbors([test_img])
    img_index = indices[0][1]

    similar_filename = old_filenames[img_index]


    ##Delete Files
    op.delete_link_img(img_name)
    op.delete_link_img(filenames_str)
    op.delete_link_img(features_str) 

    img_url = op.find_document(similar_filename,'_id')[0]['url']
    return img_url
    


    #return Filename
#    if distances[0][1] < 0.51:
#        return similar_filename   #or URL
#    else:
#        return None
#fil = test_similarity('idJ1p1GSLvQIXinkQl1Ic51jsJk2','697cebb2-7e2b-419f-ba17-1cda8d7778f2.jpg')
#print("Name: ",fil)
    