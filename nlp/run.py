import streamlit as st
from reviews.utils import PredictSentiment
#import reviews

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
            #high=-10
            st.header(string)
        else: 
            string = "{}% Positive!:thumbsup:".format(round(prob,2))
            cmap='Blues'
            #high=10
            st.header(string)

if app=='Article classification':
    st.header('Article classification')