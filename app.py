import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
from PIL import Image


pickle_in = open("Xgboost.pkl","rb")
classifier=pickle.load(pickle_in)

st.set_page_config(layout="wide")

def welcome():
    return "Welcome All"

def predict(radius_mean, texture_mean, perimeter_mean, area_mean,
       smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, perimeter_se, area_se, smoothness_se,
       compactness_se, concavity_se, concave_points_se, symmetry_se,
       fractal_dimension_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst):
    
    prediction= classifier.predict([[radius_mean, texture_mean, perimeter_mean, area_mean,
       smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, perimeter_se, area_se, smoothness_se,
       compactness_se, concavity_se, concave_points_se, symmetry_se,
       fractal_dimension_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst]])
    print(prediction)
    return prediction



def main():
    html_temp = """
    <div style="background-color:grey;padding:10px">
    <h2 style="color:white;text-align:center;"> Benign and Malign Cancer Classification Application </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        st.markdown("### Mean Values")
        radius_mean = st.number_input("radius_mean", min_value=6.981, max_value=28.11, help="Value must lie between 6.981 and 28.11")
        texture_mean = st.number_input("texture_mean", min_value=9.71, max_value=39.28, help="Value must lie between 9.71 and 39.28")
        perimeter_mean = st.number_input("perimeter_mean", min_value=43.79, max_value=188.5, help="Value must lie between 43.79 and 188.5")
        area_mean = st.number_input("area_mean", min_value=143.5, max_value=2501.0, help="Value must lie between 143.5 and 2501.0")
        smoothness_mean = st.number_input("smoothness_mean", min_value=0.05263, max_value=0.1634, help="Value must lie between 0.05263 and 0.1634")
        compactness_mean = st.number_input("compactness_mean", min_value=0.01938, max_value=0.3454, help="Value must lie between 0.01938 and 0.3454")
        concavity_mean = st.number_input("concavity_mean", min_value=0.0, max_value=0.4268, help="Value must lie between 0.0 and 0.4268")
        concave_points_mean = st.number_input("concave_points_mean", min_value=0.0, max_value=0.2012, help="Value must lie between 0.0 and 0.2012")
        symmetry_mean = st.number_input("symmetry_mean", min_value=0.106, max_value=0.304, help="Value must lie between 0.106 and 0.304")
        fractal_dimension_mean = st.number_input("fractal_dimension_mean", min_value=0.04996, max_value=0.09744, help="Value must lie between 0.04996 and 0.09744")

    # Second column inputs
    with col2:
        st.markdown("### SE Values")
        radius_se = st.number_input("radius_se", min_value=0.1115, max_value=2.873, help="Value must lie between 0.1115 and 2.873")
        texture_se = st.number_input("texture_se", min_value=0.3602, max_value=4.885, help="Value must lie between 0.3602 and 4.885")
        perimeter_se = st.number_input("perimeter_se", min_value=0.757, max_value=21.98, help="Value must lie between 0.757 and 21.98")
        area_se = st.number_input("area_se", min_value=6.802, max_value=542.2, help="Value must lie between 6.802 and 542.2")
        smoothness_se = st.number_input("smoothness_se", min_value=0.001713, max_value=0.03113, help="Value must lie between 0.001713 and 0.03113")
        compactness_se = st.number_input("compactness_se", min_value=0.002252, max_value=0.1354, help="Value must lie between 0.002252 and 0.1354")
        concavity_se = st.number_input("concavity_se", min_value=0.0, max_value=0.396, help="Value must lie between 0.0 and 0.396")
        concave_points_se = st.number_input("concave_points_se", min_value=0.0, max_value=0.05279, help="Value must lie between 0.0 and 0.05279")
        symmetry_se = st.number_input("symmetry_se", min_value=0.007882, max_value=0.07895, help="Value must lie between 0.007882 and 0.07895")
        fractal_dimension_se = st.number_input("fractal_dimension_se", min_value=0.000895, max_value=0.02984, help="Value must lie between 0.000895 and 0.02984")

    # Third column inputs
    with col3:
        st.markdown("### Worst Values")
        radius_worst = st.number_input("radius_worst", min_value=7.93, max_value=36.04, help="Value must lie between 7.93 and 36.04")
        texture_worst = st.number_input("texture_worst", min_value=12.02, max_value=49.54, help="Value must lie between 12.02 and 49.54")
        perimeter_worst = st.number_input("perimeter_worst", min_value=50.41, max_value=251.2, help="Value must lie between 50.41 and 251.2")
        area_worst = st.number_input("area_worst", min_value=185.2, max_value=4254.0, help="Value must lie between 185.2 and 4254.0")
        smoothness_worst = st.number_input("smoothness_worst", min_value=0.07117, max_value=0.2226, help="Value must lie between 0.07117 and 0.2226")
        compactness_worst = st.number_input("compactness_worst", min_value=0.02729, max_value=1.058, help="Value must lie between 0.02729 and 1.058")
        concavity_worst = st.number_input("concavity_worst", min_value=0.0, max_value=1.252, help="Value must lie between 0.0 and 1.252")
        concave_points_worst = st.number_input("concave_points_worst", min_value=0.0, max_value=0.291, help="Value must lie between 0.0 and 0.291")
        symmetry_worst = st.number_input("symmetry_worst", min_value=0.1565, max_value=0.6638, help="Value must lie between 0.1565 and 0.6638")
        fractal_dimension_worst = st.number_input("fractal_dimension_worst", min_value=0.05504, max_value=0.2075, help="Value must lie between 0.05504 and 0.2075")

    

    result = ""
    if st.button("Predict"):
        result= predict(radius_mean, texture_mean, perimeter_mean, area_mean,
       smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, perimeter_se, area_se, smoothness_se,
       compactness_se, concavity_se, concave_points_se, symmetry_se,
       fractal_dimension_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst)
        if(result[0]):        
            st.error("It is a malignant (cancerous) tumour")
        else:
            st.success("It is a benign (non-cancerous) tumour")



  

if __name__=='__main__':
    main()
    
    
