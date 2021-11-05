import sys,os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from pickle import dump, load
import joblib

def tt_split(X, labels):
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def loadData(path):
        """
        Loads the data from the data folder and builds the dataframe.
        Returns
            DataFrame:
        """
        data_path=os.path.join(pathlib.Path().absolute(), path, 'spam.csv')
        df_spam = pd.read_csv(data_path, encoding='ISO-8859-1')
        df_spam = df_spam.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        df_spam.columns = ['labels', 'data']
        df_spam['binary_labels'] = df_spam['labels'].map({'ham':0, 'spam':1})

        return df_spam

class SpamTrain(object):
    def __init__(self):
        pass

    def train(self):
        
        df_spam = loadData('data')
        x_train, x_test, y_train, y_test = tt_split(df_spam['data'], df_spam['binary_labels'])

        text_clf = Pipeline([('tfidf', TfidfVectorizer()),\
                     ('clf', SVC(kernel='linear', C=2.))])

        #text_clf.fit(df_spam['data'], df_spam['binary_labels'])
        text_clf.fit(x_train, y_train)

        model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/svc_tfidf.pkl"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        dump(text_clf, open(model_file, 'wb'))

class SpamAnalyse(object):
    def __init__(self):
        model_path = os.path.join(str(pathlib.Path().absolute()), "spam/model")
        #model_path = os.path.join(str(pathlib.Path().absolute()), "model")
        model_file = model_path + "/svc_tfidf.pkl"
        self.model = joblib.load(model_file)


    def analyseModel(self):
        """Loads the model and predicts with test data. 
        Args:
        Returns:
            ndarray: containing the confusion matrix of the predicted values vs the true labels. 
        """
        
        df_spam = loadData('spam/data')
        #df_spam = loadData('data')
        x_train, x_test, y_train, y_test = tt_split(df_spam['data'], df_spam['binary_labels'])
        y_pred = self.model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_df = pd.DataFrame([recall, prec, acc, f1], index = ['recall','precision','accuracy','f1 score'], columns=['Score'])

        return cm, metrics_df

class SpamPredict(object):
    def __init__(self):
        try:
            sys.path.index(os.path.join(str(pathlib.Path().absolute()), "spam"))
        except ValueError:
            sys.path.append(os.path.join(str(pathlib.Path().absolute()), "spam"))

        model_path = os.path.join(str(pathlib.Path().absolute()), "spam/model")
        model_file = model_path + "/svc_tfidf.pkl"
        self.model = joblib.load(model_file)


    def predict(self, text):
        pred = self.model.predict([text])

        return pred[0]
