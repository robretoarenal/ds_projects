import streamlit as st
from housing.utils_housing import HousePredict

def housePrices():
    st.title('Housing price!')
    st.write('An app to know how much a house in California would cost')

    with st.sidebar.expander('Input variables for house prediction'):
        #st.sidebar.header('Input variables from model for house prediction')
        st.subheader('Input Longitude')
        st.write('Beware that the model was built for California. A good default is -118')
        longitude = st.slider('Select longitude', -180.0, 180.0, -118.0)

        st.subheader('Input Latitude')
        st.write('Beware that the model was built for California. A good default is 34')
        latitude = st.slider('Select latitude', 0.0, 90.0, 34.0)

        st.subheader('Input Housing Median Age')
        st.write('Beware that the model was built for California. A good default is 18')
        housing_median_age = st.slider('Select the median age of housing', 0, 100, 18)

        st.subheader('Input Total Rooms in Neighbourhood')
        st.write('Beware that the model was built for California. A good default is 3700')
        total_rooms = st.slider('Select the rooms in the neighbourhood', 0.0, 10000.0, 3700.0)

        st.subheader('Input Total Bedrooms in Neighbourhood')
        st.write('Beware that the model was built for California. A good default is 400')
        total_bedrooms = st.slider('Select the total bedrooms in neighbourhood', 0.0, 10000.0, 400.0)

        st.subheader('Input Population')
        st.write('Beware that the model was built for California. A good default is 3300')
        population = st.slider('Select total population in neighbourhood', 0.0, 10000.0, 3300.0)

        st.subheader('Input Total Households')
        st.write('Beware that the model was built for California. A good default is 1400')
        households = st.slider('Select total households per neighbourhood', 0.0, 10000.0, 1400.0)

        st.subheader('Input Median Income')
        st.write('Beware that the model was built for California. A good default is 2')
        median_income = st.slider('Select the median income per month (in thousands)', 0.0, 100.0, 2.0)

        st.subheader('Input Ocean Proximity')
        st.write('You can only choose those options')
        ocean_distance = st.selectbox('Select house ocean distance', options = ['<1H OCEAN', 'NEAR OCEAN', 'INLAND'])

    st.subheader('Selected values')
    pred, df = HousePredict().predict(longitude=longitude,
                                  latitude=latitude,
                                  housing_median_age=housing_median_age,
                                  total_rooms=total_rooms,
                                  total_bedrooms=total_bedrooms,
                                  population=population,
                                  households=households,
                                  median_income=median_income,
                                  ocean_proximity=ocean_distance)

    if st.checkbox("Show input DataFrame:"):
        st.write('The values that you have selected are')
        st.write(df.head())
    
    st.header('PREDICTED HOUSE PRICE!')
    st.write("Your selection of inputs result in a price of ", pred[0], " !!")

    st.map(df)

def main():
    st.set_page_config(layout="wide")
    st.sidebar.header('Choose the Machine learning application:')
    values = ['<select>','House prices']
    app = st.sidebar.selectbox('Select:',values)

    if app == 'House prices':
        housePrices()

if __name__ == "__main__":
    main()
