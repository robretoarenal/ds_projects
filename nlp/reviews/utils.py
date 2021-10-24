import os
import pathlib
#import tarfile
import urllib.request
import pandas as pd
import spacy
import string
#import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pickle import dump, load
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


    
#Custom transformer using Python standard library (you could use spacy as well)
class predictors(TransformerMixin):
    """Class used to perform the first step of pipeline. 
    This consists in lower case all sentences. 
    
    """
    # This function will clean the text
    def clean_text(self,text):     
        return text.strip().lower()

    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]
        #return [text for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
        

class SentimentTrain(object):
    """ Class used to train the sentiment analysis model

    Attributes:
        data_path (str): path where the text files can be found.
    """

    def __init__(self,data_path):
        self.data_path=os.path.join(pathlib.Path().absolute(), data_path)

    def prepareData(self):
        """ Method that read each txt file and joins them. 
        Returns:
            DataFrame: Including the joined files with columns 'Message' and 'Target'
        
        """

        df_yelp = pd.read_table(os.path.join(self.data_path,'yelp_labelled.txt'))
        df_imdb = pd.read_table(os.path.join(self.data_path,'imdb_labelled.txt'))
        df_amz = pd.read_table(os.path.join(self.data_path,'amazon_cells_labelled.txt'))
        # Concatenate our Datasets
        frames = [df_yelp,df_imdb,df_amz]

        for column in frames: 
            column.columns = ["Message","Target"]

        df_reviews = pd.concat(frames)
        return df_reviews

    def spacy_tokenizer(self,doc):
        """Function that serves as tokenizer in our pipeline
        Loads the 'en_core_web_sm' model, tokenize the string and perform pre processing. 
        Preprocessing includes lemmatizing tokens as well as removing stop words and punctuations. 
        Args:
            doc(str): sentence to tokenize.
        Returns: 
            list: preprocessed tokens. 
        """
        punctuations = string.punctuation
        nlp = spacy.load('en_core_web_sm')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        tokens = nlp(doc)

        # Lemmatizing each token and converting each token into lowercase
        tokens = [word.lemma_.lower() for word in tokens if not word.is_space]        
        # Removing stop words and punctuations
        tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]
        # return preprocessed list of tokens
        return tokens

    def train(self):
        """Function that performs a pipeline execution.

        This function creates a Pipeline instance. Splits the data into train/test and pass it through the pipeline. 
        It also saves the model as pickle file once training is over. 

        """
        df_reviews = self.prepareData()

        tfvectorizer = TfidfVectorizer(tokenizer = self.spacy_tokenizer)
        classifier_LG = LogisticRegression(verbose=True)

        pipe2_LG = Pipeline([
            ('vectorizer', tfvectorizer),
            ('classifier', classifier_LG)], verbose=True)

        # pipe2_LG = Pipeline([
        #     ("cleaner", predictors()),
        #     ('vectorizer', tfvectorizer),
        #     ('classifier', classifier_LG)], verbose=True)

        X = df_reviews['Message']
        ylabels = df_reviews['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)
        pipe2_LG.fit(X_train,y_train)

        # Save the model
        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/logreg_tfidf.pkl"

        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        dump(pipe2_LG, open(model_file, 'wb'))


class PredictSentiment(object):
    """ Class to load the model and build the tokens DataFrame
    This class will load the model using the pickle file. So it can be used by the predict method.
    
    """

    def __init__(self):
        #model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        #model_file = model_path + "/forest_reg.pkl"
        #self.model = load(open(model_file, 'rb'))
        self.model = joblib.load("model/logreg_tfidf.pkl")

    def buildDF(self, sentence):
        """Generate DataFrame with tokens and coefficients.
        Args:
            sentence(str): sentence to tokenize. 
        Returns:
            DataFrame: containing tokens used for prediction and corresponding coeficients,
        
        """

        tokens = SentimentTrain("Data").spacy_tokenizer(sentence[0])
        arr=[]
        for token in tokens:
            idx = self.model.steps[1][1].vocabulary_.get(token)
            coef = self.model.steps[2][1].coef_[0][idx]
            arr.append({'TOKEN':token, 'Coef':coef})

        return pd.DataFrame(arr)



    def predict(self, sentence):
        """ Calls the predict and predict_proba function of the model. 

        Args: 
            sentence(str) to predict sentiment. 
        
        Returns: 
            int- predicted class. 0 if it is negative sentiment or 1 if it is positive.
            float- probability of being part of the predicted class. 
            DataFrame- DataFrame with the tokens used to predict and their coeficients.

        """

        predict = self.model.predict(sentence)
        pred_prob = self.model.predict_proba(sentence)
        prob=0
        if predict[0] == 0:
            prob = pred_prob[0][0] * 100
        else: 
            prob = pred_prob[0][1] * 100

        #tokens = SentimentTrain("Data").spacy_tokenizer(sentence[0])
        df_tokens = self.buildDF(sentence)

        return predict, prob, df_tokens

class GetData(object):
    """ Class to load the data fileted by positive and negative sentences.

    Attributes: 
        data_path: path where the text files can be found.

    """
    def __init__(self,data_path):
        self.data_path=os.path.join(pathlib.Path().absolute(), data_path)

    def dataLoad(self,dataset):
        """
        Args: 
            dataset(int): number of selected dataset, 1=Yelp 2=IMDB 3=Amazon
        Returns:
            DataFrame: First one is the positive sentences, Sencond one is the negative sentences. 

        """
        if dataset==1:
            df = pd.read_table(os.path.join(self.data_path,'yelp_labelled.txt'))
            
        elif dataset==2:
            df = pd.read_table(os.path.join(self.data_path,'imdb_labelled.txt'))
            
        else:
            df = pd.read_table(os.path.join(self.data_path,'amazon_cells_labelled.txt'))

        df.columns = ["Message","Target"]

        return df[df["Target"]==1], df[df["Target"]==0]








