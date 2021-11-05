import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt   

from reviews.utils import PredictSentiment
import reviews.utils

from articles.utils_articles import countVec,lemmatize, TopicPredict

from spam.utils_spam import SpamPredict, SpamAnalyse

def revAnalysis():
    st.header('Review Analysis')

    st.markdown("""
        This application contains a trained sentiment analysis model, capable of telling 
        whether a sentence is positive or negative. The data used to develop this model comes from 3 different
        sources, all of them being customers' opinion about the service/product they consumed.""")

    user_input = st.text_input("Write your sentence: ", ' ')
    inp_arr = [user_input]
    if st.button('Predict'):
        #pred,prob,df_tokens=utils.PredictSentiment().predict(inp_arr)
        pred,prob,df_tokens=PredictSentiment().predict(inp_arr)
        st.header(user_input)
        if pred[0]==0:
            string = "{}% Negative!:thumbsdown:".format(round(prob,2))
            cmap='OrRd'
            df_tokens.Coef = df_tokens.Coef * -1
            st.header(string)
        else: 
            string = "{}% Positive!:thumbsup:".format(round(prob,2))
            cmap='Blues'
            st.header(string)

def artClass():
    """ 
    """

    st.title('Article classification')
    #st.header('Article classification')
    
    st.markdown("""
        This application reads an article of your preference and predicts its class. 
        The application use a random forest model trained with 2500 articles from BBC.
        
        The possible classes are: entretainment, politics, sports, business, and tech.""")

    st.write("The dataset used to train this model was retrieved from http://mlg.ucd.ie/datasets/bbc.html")

    uploaded_file = st.file_uploader("Upload your article HERE:",type=['txt'])
    if uploaded_file is not None:
        with st.form(key='my_form'):
            encoding = 'utf-8'
            art = str(uploaded_file.read(), encoding)
            text_input = st.text_area(label='Loaded article',value = art, height=250)
            submit_button = st.form_submit_button(label='Process')

        if submit_button:
            #file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            #st.write(file_details)
            
            pred, prob, prob_df, imp_df = TopicPredict().predict(art)
            st.header("The article's topic is: ")
            st.text(pred[0] + '(' + str(prob*100) + '%)')
            
            col1, col2 = st.columns(2)
            col1.header("Class probabilities")
            col1.table(prob_df)
            col2.header("Top 10 Features")
            col2.table(imp_df)
            #st.write(imp_df)

def spamDetection():

    st.title('Article classification')
    st.markdown("""
    This app tells if the text provided is prone to be spam or not spam. 

    """)
    

    my_expander = st.expander(label='Inspect Model', expanded=False)
    with my_expander:
        st.text(""" The data set contains 5572 labeled emails. Being 4825 Not spam and 747 spam.""")
        st.text("The model is a supporting vector machine trained with 80% of data and tested with the rest 20%.")
        st.text("As preprocesing step, TfIdf was used before training the model.")
        clicked = st.button("View model's performance")
        if clicked:
            sa = SpamAnalyse()
            cm, df = sa.analyseModel()
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='g', ax=ax)
            
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['Not spam', 'Spam'])
            ax.yaxis.set_ticklabels(['Not spam', 'Spam'])

            col1, col2 = st.columns(2)
            col1.write(fig)
            col2.write('Metrics: ')
            col2.table(df)
            #col2.write(df)
            
    with st.form(key = 'my_form'):
        text = st.text_area(label = "Paste the email content here:", height=250)
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        pred = SpamPredict().predict(text)
        if pred==0:
            st.header("This text is predicted to be NOT SPAM")
        
        if pred==1:
            w = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">WARNING!!</p>'
            st.markdown(w, unsafe_allow_html=True)
            st.header("This text is predicted to be SPAM")



def main():
    st.set_page_config(layout="wide")
    st.sidebar.header('Choose the NLP application:')
    app = st.sidebar.selectbox('Select:',['Review Analysis','Article Classification','Spam Detection'])

    if app == 'Review Analysis':
        revAnalysis()
    
    if app == 'Article Classification':
        artClass()

    if app == 'Spam Detection':
        spamDetection()


if __name__ == "__main__":
    main()





