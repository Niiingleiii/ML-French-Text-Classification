# -*- coding: utf-8 -*-
"""UI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bIjymU_l5oZw7EwksoKFOID_DPpojskW
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
from transformers import CamembertTokenizer, CamembertModel
import lightgbm as lgb
from datasets import Dataset
import traceback

# Load the models and tokenizer
model_path = '/Users/leining/Desktop/camembert_finetuned_full/camembert_finetuned'
lgb_model_path = '/Users/leining/Desktop/camembert_finetuned_full/lgb_model_best.txt'
additional_info_path = '/Users/leining/Desktop/camembert_finetuned_full/additional_info.json'

try:
    # Load LightGBM model
    lgb_model_best = lgb.Booster(model_file=lgb_model_path)

    # Load Camembert model and tokenizer
    model = CamembertModel.from_pretrained(model_path)
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    
    # Load additional information from JSON
    with open(additional_info_path, 'r') as f:
        additional_info = json.load(f)
    
    num_characters_mean = additional_info['num_characters_mean']
    num_characters_std = additional_info['num_characters_std']
    average_length_mean = additional_info['average_length_mean']
    average_length_std = additional_info['average_length_std']

    # Difficulty mapping
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Define the prediction function
    def predict_difficulty(text):
        # Preprocess the text
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            sentence_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        num_characters = len(text)
        average_length = np.mean([len(word) for word in text.split()])
        
        # Normalize features
        num_characters = (num_characters - num_characters_mean) / num_characters_std
        average_length = (average_length - average_length_mean) / average_length_std
        
        # Combine features
        combined_features = np.concatenate((sentence_features, [[num_characters, average_length]]), axis=1)
        
        # Predict difficulty
        y_pred = lgb_model_best.predict(combined_features)
        return y_pred[0]

    # Define the Streamlit layout
    st.title('Text Difficulty Prediction App')
    st.write('This application predicts the difficulty level of French sentences. You can upload a CSV file or input sentences directly.')

    # Tab layout for file upload and text input
    tab1, tab2 = st.tabs(["Upload CSV", "Input Sentence"])

    with tab1:
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'sentence' in data.columns:
                st.write('Data successfully loaded!')
                with st.spinner('Predicting...'):
                    # Processing and prediction
                    data['sentence'] = data['sentence'].astype(str)
                    predictions = [predict_difficulty(sentence) for sentence in data['sentence']]
                    data['predicted_difficulty'] = [difficulty_mapping[int(i)] for i in predictions]

                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'predicted_difficulty']])
                    st.download_button(label='Download Predictions', data=data.to_csv().encode('utf-8'), file_name='predicted_difficulties.csv', mime='text/csv')
            else:
                st.error('Uploaded file does not contain required "sentence" column.')

    with tab2:
        st.header("Input Sentence Directly")
        sentence = st.text_area("Enter the sentence here:")
        if st.button("Predict Difficulty"):
            if sentence:
                with st.spinner('Predicting...'):
                    # Process the sentence
                    predicted_difficulty = predict_difficulty(sentence)
                    st.success(f'The predicted difficulty level for the input sentence is: {difficulty_mapping[int(predicted_difficulty)]}')
            else:
                st.error("Please enter a sentence for prediction.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    traceback.print_exc()
