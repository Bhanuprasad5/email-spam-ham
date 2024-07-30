import pickle

import pandas as pd
import sklearn
from PIL import Image
from sklearn.naive_bayes import MultinomialNB

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


#image= Image.open(r"573326-innomatics_research_labs_logo.png")
#st.image(image)
# st.image(r"573326-innomatics_research_labs_logo.png")
model = pickle.load(open(r"email spam ham.pkl","rb"))
tf = pickle.load(open(r"vectorization.pkl","rb"))
email = st.text_input("Enter the Email:")





if st.button("Submit"):
    
    data = tf.transform([email]).toarray()
    
    pred = model.predict(data)[0]
    st.write("The Email is : ",pred)
