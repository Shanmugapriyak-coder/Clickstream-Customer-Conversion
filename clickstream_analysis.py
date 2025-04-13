import streamlit as st 
import pandas as pd
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import bz2file as bz2
import pickle

from custom_function import (
    replace_country_func,
    replace_page1_func,
    replace_colour_func,
    replace_location_func,
    replace_model_photo_func,
    replace_clothing_photo_func,
    filter_countries_func
)
from custom_function import country_mapping, page1, colors,locations,model_photo,clothing_model

def decompress_pickle(file):
    data=bz2.BZ2File(file,'rb')
    data=pickle.load(data)
    return data

regresion_model = joblib.load(r"regression_model.pkl")
classification_model = joblib.load(r"classification_model.pkl")

classification_model=decompress_pickle('classification_model.pbz2')
def intro():
    
    st.write("# Welcome to Click Stream Customer Conversion👋")
    st.sidebar.error("Select a service above. 🙂 ")

    st.markdown(
        """ In today’s highly competitive e-commerce landscape, understanding customer behavior is key to 
        boosting engagement and maximizing revenue. As a data scientist at a leading e-commerce platform, 
        I have developed an intelligent and user-friendly Streamlit web application that leverages 
        clickstream data to drive actionable business insights.

        This interactive application is designed to solve three core problems in the e-commerce space:
        🔍 1. Classification - Purchase Prediction
        💰 2. Regression - Revenue Estimation
        🧠 3. Clustering - Customer Segmentation
        **👈 Select a service from the sidebar**

       
    """
    )


def revenue_form():
    st.title("Revenue Estimation")
    form = st.form(key='registration_form')
    year = 2008
    month = form.number_input("month", min_value=4, max_value=8, step=1)
    day = form.number_input("day ", min_value=1, max_value=31, step=1)
    order = form.number_input("order", min_value=1, max_value=200, step=1)
    country = form.selectbox("Choose your country",('--None--','Poland', 'Czech Republic', 'Lithuania', 'net (.net)', 'com (.com)','Others'))
    page1_main_category = form.selectbox("Choose your main  category",('--None--','trousers','skirts','blouses','sale'))
    page2_clothing_model = form.selectbox("Choose your Clothing model",('--None--','C', 'B', 'A', 'P'))
    colour = form.selectbox("Choose your colour",('--None--','violet', 'of many colors', 'black', 'red', 'pink', 'blue',
       'white', 'green', 'brown', 'beige', 'gray', 'navy blue', 'olive','burgundy'))
    location = form.selectbox("Choose your colour",('--None--','top left', 'top right', 'bottom in the middle', 'bottom left',
       'top in the middle', 'bottom right'))
    model_photography = form.selectbox("Choose your model photographyr",('--None--','profile', 'en face'))
       
    page= form.number_input("Page", min_value=1, max_value=5, step=1)
    submit_button = form.form_submit_button(label='Predict price')
    
    if submit_button  :
            df=pd.DataFrame({'year':[year],'month':[month],'day':[day],
                        'order':[order],'country':[country],
                        'page1_main_category':[page1_main_category],'page2_clothing_model':[page2_clothing_model],
                        'colour':[colour],'location':[location],'model_photography':[model_photography],
                        'page':[page]})

          
            price=regresion_model.predict(df)
            st.write("Predicted price is ",price[0])

def Purchase_Prediction_form():
    st.title("Revenue Estimation")
    form = st.form(key='registration_form1')
    year = 2008
    month = form.number_input("month", min_value=4, max_value=8, step=1)
    day = form.number_input("day ", min_value=1, max_value=31, step=1)
    order = form.number_input("order", min_value=1, max_value=200, step=1)
    country = form.selectbox("Choose your country",('--None--','Poland', 'Czech Republic', 'Lithuania', 'net (.net)', 'com (.com)','Others'))
    page1_main_category = form.selectbox("Choose your main  category",('--None--','trousers','skirts','blouses','sale'))
    page2_clothing_model = form.selectbox("Choose your Clothing model",('--None--','C', 'B', 'A', 'P'))
    colour = form.selectbox("Choose your colour",('--None--','violet', 'of many colors', 'black', 'red', 'pink', 'blue',
       'white', 'green', 'brown', 'beige', 'gray', 'navy blue', 'olive','burgundy'))
    location = form.selectbox("Choose your colour",('--None--','top left', 'top right', 'bottom in the middle', 'bottom left',
       'top in the middle', 'bottom right'))
    model_photography = form.selectbox("Choose your model photographyr",('--None--','profile', 'en face'))
       
    page= form.number_input("Page", min_value=1, max_value=5, step=1)
    submit_button = form.form_submit_button(label='Predict Purchase')
    
    if submit_button  :
            df=pd.DataFrame({'year':[year],'month':[month],'day':[day],
                        'order':[order],'country':[country],
                        'page1_main_category':[page1_main_category],'page2_clothing_model':[page2_clothing_model],
                        'colour':[colour],'location':[location],'model_photography':[model_photography],
                       'page':[page]})

          
            result=classification_model.predict(df)
            # st.write("Predicted price is ",price[0])
            st.subheader("🎯 Prediction Results")
            if result[0]==1 :
                 st.metric(label="", value=" Will Purchase the Product")
            else:
                
                st.markdown("### ❌ The customer is **unlikely to make a purchase**.")

def cluster_form():
    
   

    st.title("🧠 Customer Segments - Clustering Visualization")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload customer data CSV", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.write(data.head())

        # Select numerical features
        num_data = data.select_dtypes(include=['float64', 'int64'])
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(num_data)
        
        # KMeans clustering
        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = clusters

        # PCA for 2D plotting
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]

        st.subheader("🟢 Cluster Visualization (2D PCA)")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
        st.pyplot(fig1)

        st.subheader("📊 Cluster Distribution - Pie Chart")
        fig2, ax2 = plt.subplots()
        data['Cluster'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'), ax=ax2)
        ax2.set_ylabel('')
        st.pyplot(fig2)

        st.subheader("📉 Cluster Distribution - Bar Chart")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='Cluster', data=data, palette='Set2', ax=ax3)
        st.pyplot(fig3)

        st.subheader("📈 Feature Distribution - Histogram")
        feature = st.selectbox("Select feature to display", num_data.columns)
        fig4, ax4 = plt.subplots()
        for c in sorted(data['Cluster'].unique()):
            sns.histplot(data[data['Cluster'] == c][feature], label=f"Cluster {c}", kde=True, ax=ax4)
        ax4.legend()
        st.pyplot(fig4)

page_names_to_funcs = {
     "--None--":intro,
    "Revenue Estimation":revenue_form,
    "Purchase Prediction":Purchase_Prediction_form,
    "cluster_form":cluster_form
}
demo_name = st.sidebar.selectbox("Choose a service", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
