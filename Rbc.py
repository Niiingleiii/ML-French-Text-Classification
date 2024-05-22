import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set page configuration at the top
st.set_page_config(page_title="Text Difficulty Level Prediction", page_icon="📚")

# Load model and vectorizer once when the app starts
@st.cache_resource
def load_model(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# File path for the model and vectorizer
file_path = "/Users/leining/Desktop/model_LR.pkl"
model_LR, vectorizer = load_model(file_path)

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Streamlit application
def main():
    st.title("Text Difficulty Level Prediction 📚")
    st.markdown("**Enter the French text below to predict its difficulty level.**")
    
    preface_text = st.text_area("Enter the text for difficulty prediction:", height=200)
    
    if st.button("Predict"):
        with st.spinner('🔍 Predicting difficulty level...'):
            preface_transformed = vectorizer.transform([preface_text])
            prediction = model_LR.predict(preface_transformed)
            st.success(f'The predicted difficulty level is: {prediction[0]}')
            
            st.subheader("Word Cloud of Your Text")
            wordcloud = generate_wordcloud(preface_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            st.subheader("Compare with Sample Texts")
            sample_texts = {
                "A1": "Le bleu, c'est ma couleur préférée mais je n'aime pas le vert!",
                "A2": "Les médecins disent souvent qu'on doit boire un verre de vin rouge après les repas.",
                "B1": "Nous allons bien, nous habitons dans une petite maison ancienne avec un très beau jardin.",
                "B2": "Certes il doit répondre aux goûts du consommateur, mais aussi aux capacités de son porte-monnaie.",
                "C1": "Les médecins des donneurs vous conseilleront volontiers dans un tel cas vous renseigneront par rapport aux dons de sang.",
                "C2": "Un revenu devient donc une nécessité pour que l'Homme puisse accéder à la satisfaction d'avoir comblé ses désirs."
            }
            for level, text in sample_texts.items():
                st.markdown(f"**{level} Text**: {text}")

if __name__ == '__main__':
    main()

