import sys,os
import pathlib
import joblib
import pandas as pd
import numpy as np
import spacy
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump, load
import string

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space

def lemmatize(doc):
    """
    function that tokenize the text, lemmatizes it and removes stop words.
    """
    nlp = spacy.load('en_core_web_sm')
    parsed_doc = nlp(doc)
    lemm_doc = [token.lemma_ for token in parsed_doc
                      if not punct_space(token) and (token.lemma_!= '-PRON-') and not(nlp.vocab[token.text].is_stop)]
      
    # write the transformed text
    clean_text = u' '.join(lemm_doc)
    return clean_text

def countVec(self, article):
    cvec = CountVectorizer(stop_words='english', min_df = 3)
    cvec.fit(article)
    cvec_counts = cvec.transform(article)
    return cvec_counts

nlp = spacy.load('en_core_web_sm')

def spacy_tokenizer(doc):
        """Function that serves as tokenizer in our pipeline
        Loads the 'en_core_web_sm' model, tokenize the string and perform pre processing. 
        Preprocessing includes lemmatizing tokens as well as removing stop words and punctuations. 
        Args:
            doc(str): sentence to tokenize.
        Returns: 
            list: preprocessed tokens. 
        """

        punctuations = string.punctuation
        
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        tokens = nlp(doc)

        # Lemmatizing each token and converting each token into lowercase
        tokens = [word.lemma_.lower() for word in tokens if not word.is_space]        
        # Removing stop words and punctuations
        tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]
        # return preprocessed list of tokens
        return tokens

class TopicTrain(object):
    def __init__(self):
        #self.data_path=os.path.join(pathlib.Path().absolute(), data_path)
        pass

    def _loadArticles(self,path):
        """Loads the articles from each folder inside the path provided. Each folder will represent
        the category(label) of the articles.  
        Args:
            path(str): path where the function will find the article subfolders. 
        Returns: 
            DataFrame: containing the articles and their category
        """
        cat_article = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if '.txt' in file:
                    category = subdir.split('/')[-1]
                    f = open(os.path.join(subdir, file),'r', encoding='utf-8', errors='ignore')
                    lines = f.readlines()
                    lines = ' '.join(lines).replace('\n','')
                    #list of lists: [category,article]
                    cat_article.append([category,lines])
                    f.close()

        data = pd.DataFrame(cat_article)
        data.columns = ['category','article']
        return data


    def train(self):
        """ Creates a pipeline and trains it. The pipeline contains one preprocessing step 
        and the trainin of a random forest model.
        """
        articles_df = self._loadArticles('data')
        #articles_df['article_lemmatized']=articles_df.article.map(lemmatize)
        nlp = spacy.load('en_core_web_sm')
        text_clf = Pipeline([('tfidf', TfidfVectorizer(tokenizer=spacy_tokenizer,min_df=3)),\
                     ('clf', RandomForestClassifier())])
        
        text_clf.fit(articles_df['article'], articles_df['category'])

        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/rm_tfidf.pkl"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        dump(text_clf, open(model_file, 'wb'))

class TopicPredict(object):
    def __init__(self):
        try:
            sys.path.index(os.path.join(str(pathlib.Path().absolute()), "articles"))
        except ValueError:
            sys.path.append(os.path.join(str(pathlib.Path().absolute()), "articles"))

        model_path = os.path.join(str(pathlib.Path().absolute()), "articles/model")
        model_file = model_path + "/rm_tfidf.pkl"
        self.model = joblib.load(model_file)

    def _important_tokens(self,inp):
        """ Creates a dataframe with the top 10 most important tokens of the input article. 
        The most important features are taken from the random forest model. 
        Args:
            inp(str): article to process.
        Returns:
            DataFrame: Top 10 tokens sorted by descending order of improtance. 
        """
        tokens = spacy_tokenizer(inp)
        arr = []
        for token in tokens:
            #get the index of the token in the model's vocabulary
            idx = self.model.steps[0][1].vocabulary_.get(token)
            if idx is not None:#Some tokens doesnt appear in the corpus. 
                importance = self.model.steps[1][1].feature_importances_[idx] 
                arr.append({'TOKEN':token, 'Importance':importance})
        
        imp_df = pd.DataFrame(arr)
        top_imp_df = imp_df.groupby(['TOKEN','Importance'], as_index = False).count().sort_values(by = 'Importance',ascending = False).set_index('TOKEN').head(10)

        return top_imp_df

    def _probDf(self, prob_arr):
        """ Generate a DataFrame with the probabilities of each class.
        Args:
            prob_arr(nparray): array of the probabilities for each class.
        Returns:
            DataFrame: dataframe with sorted probabilities 
        """
        #proba = self.model.predict_proba([inp])
        proba_df = pd.DataFrame(prob_arr[0], index = self.model.classes_, columns = ['Proba'])

        return proba_df.sort_values(by = 'Proba', ascending = False)

    def predict(self,inp):
        """Calls the prediction function of the model. 
        Args:
            inp(str): article to predict
        Returns:
            pred(list): list with the string of the predicted category. 
            prob(float): probability of the predicted category. 
            prob_df(DataFrame): contains the probabilities of all categories.  
            imp_df(DataFrame): contains the top 10 most important tokens in the article. 
        """
        pred = self.model.predict([inp])
        prob_arr = self.model.predict_proba([inp])

        idx = np.where(self.model.classes_ == pred)[0][0]
        prob = prob_arr[0][idx]
        prob_df = self._probDf(prob_arr)
        imp_df = self._important_tokens(inp)

        return pred, prob, prob_df, imp_df
        