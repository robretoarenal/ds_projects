import streamlit as st
from reviews.utils import PredictSentiment
import reviews.utils
#import reviews

from articles.utils_articles import countVec,lemmatize, TopicPredict

st.sidebar.header('Choose the NLP application:')
app = st.sidebar.selectbox('Select:',['Review Analysis','Article classification'])

if app=='Review Analysis':
    st.header('Review Analysis')

    st.markdown("""
        This application contains a trained sentiment analysis model, capable of telling 
        whether a sentence is positive or negative. The data used to develop this model comes from 3 different
        sources, all of them being customers' opinion about the service/product they consumed.""")

    #st.text('Take a look at the different datasets:')
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

if app=='Article classification':

    st.header('Article classification')
    
    st.markdown("""
        This application reads an article of your preference and predicts its class. 
        The possible classes are: entretainment, politics, sports, business, and tech.""")
    
    uploaded_file = st.file_uploader("Upload your article HERE:",type=['txt'])
    if st.button("Process"):
        if uploaded_file is not None:
            #file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            #st.write(file_details)
            encoding = 'utf-8'
            art = str(uploaded_file.read(), encoding)
            lemma_art=lemmatize(art)
            st.write(lemma_art)

    # if st.button('Predict') and uploaded_file is not None:
    #     TopicPredict()





