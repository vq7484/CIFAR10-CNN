import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
 
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('keras_cifar10_trained_model.h5', compile=False)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):
    
        size = (180,180)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_resize = cv2.resize(image, dsize=(32, 32))
        
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    class_labels = {
      0: "Airplane",
      1: "Automobile",
      2: "Bird",
      3: "Cat",
      4: "Deer",
      5: "Dog",
      6: "Frog",
      7: "Horse",
      8: "Ship",
      9: "Truck"
    }

    for i in range(10):
        if predictions[0][i] == 1:
            st.write(f"Prediction: {class_labels[i]}")