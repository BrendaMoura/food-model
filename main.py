import numpy as np
import streamlit as st
import keras
from keras.preprocessing import image
import io

st.title('Modelo de Classificação de Comidas')


#Input types
uploaded_file = st.file_uploader('File Uploader')

img_file_buffer = st.camera_input('Camera')

#Model
def load_model():
    model = keras.models.load_model("final-food-model.h5")
    return model

modelPrediction = load_model()

#Classes
specific_classes =  ['baby_back_ribs','baklava','beef_carpaccio','bruschetta',\
                    'beet_salad','beignets','breakfast_burrito','donuts','churros','fried_rice']

#Prediction for upload file
if uploaded_file is not None:
    # Pre processamento
    img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = modelPrediction.predict(img_array)

    # Label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    st.write("Index: ", predicted_class_index)
    predicted_class = specific_classes[predicted_class_index]

    # Print
    st.write("Predicted class:", predicted_class)


#Prediction for image from camera
if img_file_buffer is not None:
    # Pre processamento
    img = image.load_img(io.BytesIO(img_file_buffer.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = modelPrediction.predict(img_array)

    # Label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = specific_classes[predicted_class_index]

    # Print 
    st.write("Predicted class:", predicted_class)