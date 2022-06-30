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
import matplotlib.pyplot as plt
from PIL import Image

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
	image = Image.open("resources/Logo.png")
	st.image(image)
	st.subheader("Climate change tweet classification 🌍")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Meet the team", "About CW1"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Meet the team":
		st.markdown("Our Team:")
		st.text("Supervisor: Claudia Elliot-Wilson")
        
		image = Image.open("resources/Mr Easy.jpg")
		st.image(image, caption='Bethuel Masango', width=182)
        
		image = Image.open("resources/Kanego.jpg")
		st.image(image, caption='Kanego Kgabalo Makhuloane', width=182)
        
		image = Image.open("resources/Madute.jfif")
		st.image(image, caption='Madute Ledwaba', width=182)
        
		image = Image.open("resources/mekayle_rajoo.jfif")
		st.image(image, caption='Mekayle Rajoo', width=182)
        
		image = Image.open("resources/moose.jfif")
		st.image(image, caption='Mosuwe Mosibi', width=182)
    
        
		image = Image.open("resources/Rodney.jpg")
		st.image(image, caption='Thabang Rodney Mabelane', width=182)
        
	if selection == "About CW1":
		st.info("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.\n\n With this context, we at CW1 have created a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.\n Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.")
		st.image("resources/Climate-change.gif", width=705)        
    
    
    
	if selection == "Information":
		st.info("Below is some information about the models as well as the data we trained these models on")
		# You can read a markdown file from supporting resources folder
		st.subheader("The various models trained:")
		model_info = st.radio('Select a model for more information:', ["Logistic Regression", "Linear Support Vector Classifier","Multinomial Naive Bayes Classifier", "AdaBoostClassifier" ])
        
		if model_info == "Logistic Regression":
			st.info("It makes use of a common S-shaped curve known as the logistic function.\nThis curve is commonly known as a sigmoid, it squeezes the range of output values to exist only between 0 and 1. \nIt has a point of inflection, which can be used to separate the feature space into two distinct areas (one for each class). \nValues greater than or equal to 0.5 are assigned to class 1 while values less than 0.5 are assigned to class 0.\n\n For further information refer to: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")
            
		if model_info == "Linear Support Vector Classifier":
			st.info("ISVM or Support Vector Machine is a linear model for classification and regression problems.\nIt can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.\n\n For further information refer to: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html")
            
		if model_info == "Multinomial Naive Bayes Classifier":
			st.info("Naive Bayes is a kind of classifier which uses the Bayes Theorem. \nIt predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class. The class with the highest probability is considered as the most likely class. \nThe multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).\n\n For further information refer to: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html")
            
		if model_info == "AdaBoostClassifier":
			st.info("An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.\n\n For further information refer to: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html")

            
		st.subheader("Raw Twitter data used:")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.info("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. Each tweet is labelled as one of the following classes:\n\n\n* 2 News: the tweet links to factual news about climate change\n * 1 Pro: the tweet supports the belief of man-made climate change\n * 0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change\n * -1 Anti: the tweet does not believe in man-made climate change") 
            
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
			fig1, ax1 = plt.subplots()
			st.subheader("Distribution of Data between our classes:")
			st.info("Below is a pie chart representing the amount of tweets found in each class, it is evident that the majority believe in man-made climate change")            
			ax1.pie([8530, 3640, 2353, 1296], explode=(0, 0.1, 0, 0), labels=['Pro', 'News','Neutral',"Anti"],autopct='%1.1f%%',shadow=True, startangle=90)
			ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
			st.pyplot(fig1)
            
			st.subheader("The most common words in each class:")
			st.info("The top 4 most occuring words accross all classes are climate, change, global and earning. The words 'rt' (retweet), 'science' and 'scientist' occur frequently, likely due to the fact that people tend to retweet the scientific studies that support their own views.\n\n Words like real, believe, think, fight, etc. occur frequently in pro climate change tweets.\n\n Anti-climate change tweets contain words such as 'liberal', 'scam', 'tax', 'liberal' and 'manmade'. We can see there is definitely a difference in language and tone between pro and anti classes.")
                    
			image = Image.open("resources/wordcloud_most.png")
			st.image(image, caption='WordCloud of the most common words in each class', width=700)
            
			st.subheader("The least common words in each class:")
			st.info("As expected, there is no overlap between the classes and these words do not provide us with anything of value.")            
			image = Image.open("resources/wordcloud_least.png")
			st.image(image, caption='WordCloud of the least common words in each class', width=700)            
            
            
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		model_option = st.selectbox('Please select a Classification Model:', ["Logistic Regression", "Linear Support Vector                 		Classifier","Multinomial Naive Bayes Classifier", "AdaBoostClassifier" ])
    
		st.info("We suggest using the Linear Support Vector Classifier as this produced the best accuracy")        
        
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type a Tweet Here")
        
		if st.button("Classify"):
			if tweet_text == "Type a Tweet Here" or tweet_text =="":
				st.success("Please enter a tweet to classify")
			else:
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
			if tweet_text != "Type a Tweet Here" and tweet_text != "":
				if prediction == 1:
					st.success("This tweet supports the belief of man-made climate change")
				if prediction == -1:
					st.success("This tweet does not believe in man-made climate change")
				if prediction == 0:
					st.success("This tweet neither supports nor refutes the belief of man-made climate change")
				if prediction == 2:
					st.success("This tweet links to factual news about climate change")
                


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
