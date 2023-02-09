# This code imports the necessary libraries and loads a pre-trained model called 'Cotton_disease_predictor.h5'.
# It then creates a file uploader in the Streamlit app for the user to upload an image.
# Once an image is uploaded, it is preprocessed by converting it to a numpy array, reshaping it to (1, 224, 224, 3) and rescaling it by dividing by 255.
# The model then makes a prediction on the preprocessed image and assigns it the label with the highest predicted probability.
# The image and its predicted label are then displayed in the Streamlit app.
# The labels list is a list of possible labels for the predictions, that being 'Plant_leaf_with_disease','Cotton_plant_with_disease','Fresh leaf','Fresh Plant'.
# The 'pred_max' variable is the index of the label with the highest predicted probability and 'pred_label' is the corresponding label.
# The 'st.image' function is used to show the uploaded image and 'st.write' function is used to display the prediction label.





from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import numpy as np
import PIL
model = load_model('Cotton_disease_predictor.h5')
st.title('Cotton Disease Prediction')


image_uploaded = st.file_uploader('Upload the image file',type=['jpg','jpeg','png'])

if image_uploaded is not None:
    image = PIL.Image.open(image_uploaded)
    img_array = np.array(image)
    reshape = np.resize(img_array,(1,224,224,3))
    # st.write(reshape.shape)
    rescale = reshape/255.
    predict = model.predict(rescale)
    pred_max = predict.argmax()
    labels = ['Plant_leaf_with_disease','Cotton_plant_with_disease','Fresh leaf','Fresh Plant']
    pred_label = labels[pred_max]
    
    st.image(image_uploaded)
    st.write(f'The prediction is {pred_label}',align='center')
    
    