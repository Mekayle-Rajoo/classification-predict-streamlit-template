"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/Vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def clean_text(mystring):
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    import re
    from nltk.tokenize import word_tokenize, TreebankWordTokenizer
    from nltk.stem import WordNetLemmatizer
    
    mystring = re.sub('http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', 'url-web', mystring)
    
    mystring = mystring.lower()
    
    without_stopwords = []
    for tweet in range(len(mystring)):
        split = mystring.split(" ")
        for word in stopwords.words('english'):
            if word in split:
                split.remove(word)
        without_stopwords.append(' '.join(map(str, split)))
    
    mystring = without_stopwords[0]
    
    mystring = mystring.translate(str.maketrans('', '', string.punctuation))
    
    lemmatizer = WordNetLemmatizer()
    
    mystring = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(mystring)])
    
        
    return mystring  



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("CW1 Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Meet the team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Meet the team":
		st.markdown("Our Team:")
		st.text("Supervisor: Claudia Elliot-Wilson")       
		st.text("Bethuel Masango")
		st.text("Kanego Kgabalo Makhuloane")
		st.text("Madute Ledwaba")
		st.text("Mekayle Rajoo")
		st.text("Mosuwe Mosibi")
		st.text("Thabang Rodney Mabelane")
    
    
    
	if selection == "Information":
		st.info("Below is some information about the models as well as the data we trained these models on")
		# You can read a markdown file from supporting resources folder
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		model_option = st.selectbox('Please select a Classification Model:', ["Logistic Regression", "Linear Support Vector                 		Classifier","Multinomial Naive Bayes Classifier", "AdaBoostClassifier" ])
        
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type a Tweet Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			tweet_text = clean_text(tweet_text)             
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_option == "Linear Support Vector                 		Classifier":
						predictor = joblib.load(open(os.path.join("resources/LSVC_model.pkl"),"rb"))
						prediction = predictor.predict(vect_text)
                        
			if model_option == "Logistic Regression":
						predictor = joblib.load(open(os.path.join("resources/LG_model.pkl"),"rb"))
						prediction = predictor.predict(vect_text)
                        
			if model_option == "Multinomial Naive Bayes Classifier":
						predictor = joblib.load(open(os.path.join("resources/MNBC_model.pkl"),"rb"))
						prediction = predictor.predict(vect_text) 
                        
			if model_option == "AdaBoostClassifier":
						predictor = joblib.load(open(os.path.join("resources/ABC_model.pkl"),"rb"))
						prediction = predictor.predict(vect_text) 
                        
                        
                        

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
