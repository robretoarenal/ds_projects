import sys,os
import pathlib
import joblib
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier

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

class TopicTrain(object):
    def __init__(self,data_path):
        self.data_path=os.path.join(pathlib.Path().absolute(), data_path)

    def _loadArticles(self,path):
        cat_article = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if '.txt' in file:
                    category = subdir.split('/')[-1]
                    f = open(os.path.join(subdir, file),'r')
                    lines = f.readlines()
                    lines = ' '.join(lines).replace('\n','')
                    #list of lists: [category,article]
                    cat_article.append([category,lines])
                    f.close()

        data = pd.DataFrame(cat_article)
        data.columns = ['category','article']
        return data


    def train(self):
        print(self.data_path)

        articles_df = self._loadArticles('data')
        articles_df['article_lemmatized']=articles_df.article.map(lemmatize)
        return articles_df
        # X_train, X_test, y_train, y_test = train_test_split(data['article_lemmatized'], data['category'], test_size=0.4, random_state=42)

        
        # text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=3)),\
        #              ('tfidf', TfidfTransformer()),\
        #              ('clf', RandomForestClassifier())])

        # text_clf.fit(X_train, y_train)
        # model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        # model_file = model_path + "/rm_tfidf.pkl"
        # if not os.path.isdir(model_path):
        #     os.makedirs(model_path)
        # dump(text_clf, open(model_file, 'wb'))

class TopicPredict(object):
    def __init__(self):
        model_path = os.path.join(str(pathlib.Path().absolute()), "articles/model")
        model_file = model_path + "/rf_tfidf.pkl"
        self.model = joblib.load(model_file)