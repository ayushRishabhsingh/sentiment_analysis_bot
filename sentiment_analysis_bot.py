import streamlit as st
import re 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import numpy as np

#load the nlp_model file and tf-idf vectorizer file
model=joblib.load('nlp_model.pkl')
tf_idf=joblib.load('tfidf.pkl')

#initialize the NLTK tool
stemmer= PorterStemmer()
stop_words=set(stopwords.words('english'))

#function to predict sentiment
def predict_sentiment(review):
    cleaned_review= re.sub('.*?','',review)
    cleaned_review = re.sub(r'[^\w\s]','',cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review =  [stemmer.stem(word) for word in filtered_review]
    tfidf_review = tf_idf.transform([''.join(stemmed_review)])
    sentiment_prediction = model.predict(tfidf_review)
    print ("sentiment_prediction", sentiment_prediction)
    predicted_class_index = np.argmax(sentiment_prediction)
    print ("predicted_class_index", predicted_class_index)
    if predicted_class_index > 0.6 : #adjust threshold as needed
        return "Positive"
    else:
        return "Negative"


#Streamlit  UI
st.title("Sentiment  Analysis")
review_to_predict = st.text_area("Enter your text here...")
if st.button("Predict statement"):
    predict_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted sentiment: ",  predict_sentiment)







