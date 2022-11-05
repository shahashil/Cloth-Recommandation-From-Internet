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


def train_resnet_model(uid,img_name):

    try:
        filenames_str = str(uid)+'-filenames.pkl'
        features_str = str(uid)+'-features.pkl'
    
        op.download_bucket_obj('pkl',filenames_str)
        op.download_bucket_obj('pkl',features_str)
    except:                                                                         #If new user then try will fail so create demo pkl
        op.upload_bucket_obj('pkl','demo-features.pkl',uid+'-features.pkl')
        op.upload_bucket_obj('pkl','demo-filename.pkl',uid+'-filenames.pkl')
    
        filenames_str = str(uid)+'-filenames.pkl'
        features_str = str(uid)+'-features.pkl'
    
        op.download_bucket_obj('pkl',filenames_str)
        op.download_bucket_obj('pkl',features_str)


    old_filenames = pickle.load(open(filenames_str,'rb'))
    old_features = pickle.load(open(features_str,'rb'))

    op.download_bucket_obj('wardrobe',img_name)

    old_filenames.append(img_name)
    old_features.append(extract_features(img_name,get_model()))

    print(old_filenames)
    print(np.array(old_features).shape)

    pickle.dump(old_filenames,open(filenames_str,'wb'))
    pickle.dump(old_features,open(features_str,'wb'))

    op.upload_bucket_obj('pkl', filenames_str, filenames_str)
    op.upload_bucket_obj('pkl', features_str, features_str)

    op.delete_link_img(img_name)
    op.delete_link_img(filenames_str)
    op.delete_link_img(features_str)    






    
