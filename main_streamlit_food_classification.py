# streamlit_app.py
import streamlit as st
import pickle
from PIL import Image
from classification_model_utils import load_model, predict

# Load class mapping from the pickle file
with open(r"/Users/sid/Downloads/Food_Image_Classification_Final/class_mapping.pkl", 'rb') as f:
    class_mapping = pickle.load(f)

# Load the pre-trained model
num_classes = len(class_mapping)  # Use the length of the class mapping
model = load_model(r"/Users/sid/Downloads/Food_Image_Classification_Final/food_classification_model.pth", num_classes)

# Streamlit app UI
st.title('🍽️ Food Image Classification')
st.markdown("""
    This application allows you to classify food images. Upload an image of food, and the model will predict the type of food it is. 
    **Supported formats**: JPG, PNG, JPEG
""")

# Section for uploading the image
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make a prediction
    st.write("Classifying...")
    predicted_label = predict(uploaded_file, model, class_mapping)  # Assume this function returns the index
    st.write(f"Predicted Label: {predicted_label}")
