# Food Image Classification Using Deep Learning (EfficientNet-B0)
This repository contains the implementation of a Food Image Classification project using the EfficientNet-B0 deep learning architecture. The project is part of the Master's capstone at the University of Queensland and aims to achieve robust classification of food images into 28 categories with high accuracy and user interactivity via a Streamlit application.

## Table of Contents
### 1. Introduction
### 2. Project Objective
### 3. Dataset Overview
### 4. Model Architecture
### 5. Key Features
### 6. Setup Instructions
### 7. Results & Analysis
### 8. Future Improvements
### 9. Acknowledgements

## 1. Introduction
With the recent rise of online food photography and computer vision advancements, this project tackles challenges in classifying food images that are affected by visual diversity, background clutter, and occlusion. By utilizing EfficientNet-B0 and Streamlit, the project builds an interactive and accurate image classification system for real-world applications like dietary tracking and automated food logging.

## 2. Project Objectives
  i) Accurate Classification: Efficiently classify food images into 28 different categories with high precision by using EfficientNet-B0.
  ii) Data Augmentation: Utilize techniques of rotation, flipping, and rescaling for better model robustness.
  iii) Transfer Learning: Leverage pre-trained models to achieve better precision and reduce overall training time.
  iv) Interactive Application: Provide real-time predictions using an easy-to-use Streamlit interface.

## 3. Dataset Overview
  i) Name: "Taste the World: A Culinary Adventure"
  ii) Source: Kaggle (open-source)
  iii) Classes: 28 food categories
  iv) Size: ~24,000 images
  v) Challenges: Class imbalance, diverse presentations, and lighting variations.

## 4. Model Architecture
EfficientNet-B0 employs compound scaling to balance network width, depth, and resolution for optimal performance. The key features include :
![Food Image Classification EfficientNet-b0](https://drive.google.com/uc?export=view&id=1kF-tD2I5RaUdnEd1MP-HG8D5pImEhQWk)

  1) Pre-trained weights from ImageNet.
  2) Dropout layers to prevent overfitting.
  3) Transfer learning for faster convergence.

Streamlit Integration
i) User-uploaded food images are classified in real-time.
ii) Displays predictions and confidence scores interactively.

## 5. Key Features
  i) Interactive Application: Upload food images and receive instant predictions.
  ii) Robust Data Processing: Includes preprocessing, augmentation, and normalization.
  iii) Fine-Tuned Model: Adapts pre-trained EfficientNet-B0 for food classification.
  iv) Visualization: Real-time insights into classification accuracy and model confidence.

## 6. Setup Instructions
  6.1) Clone the repository:
      git clone <repository-url>
      cd <repository-directory>
  6.2) Install dependencies:
      pip install -r requirements.txt
  6.3) Run the streamlit application :
      streamlit run /Users/sid/Downloads/Food_Image_Classification_Final/main_streamlit_food_classification.py
  6.4) Upload a food image to get predictions.

## 7. Results & Analysis
  i) Accuracy: Achieved ~86% validation accuracy.
  ii) Performance: Model performs well on common dishes but struggles with less-represented categories due to class imbalance.

## 8. Future Improvements
  i) Address class imbalance by augmenting underrepresented categories.
  ii) Explore advanced regularization techniques to reduce overfitting.
  iii) Enhance the application by including confidence scores and similar food suggestions.

## 9. Acknowledgements
  i) Supervisor: Dr. Xin Guo, School of Information Technology and Electrical Engineering, University of Queensland.
  ii) Dataset: Kaggle (Taste the World: A Culinary Adventure : https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset).
  iii) Frameworks Used: Python, Matplotlib, PyTorch, Streamlit.
  iv) https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
  

