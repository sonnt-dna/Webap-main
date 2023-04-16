#import libraries

import lightgbm as lgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import bz2file as bz2
import altair as alt

from Antuns.prediction_LGBM import Prediction_LGBM
from Antuns.p1_import_csv import upload_csv
from Antuns.p2_plotting_data import curve_plot
from Antuns.page_setting import page_intro

page_intro()
#-----------------------------------------------------------------
#ignore version warnings
import warnings
warnings.filterwarnings("ignore")

#Global variables
THRESHOLD_GLOBAL = 0.75

#All wells to be trained
wells_name = ["01-97-HXS-1X", "15-1-SN-1X", "15-1-SN-2X", "15-1-SN-3XST", "15-1-SN-4X", "15-1-SNN-1P",
              "15-1-SNN-2P", "15-1-SNN-3P", "15-1-SNN-4P", "15-1-SNS-1P", "15-1-SNS-2P", "15-1-SNS-4P"]
#Obtain data and label of wells
name_features = ["GR", "LLD", "LLS", "NPHI", "RHOB", "DTC", "DTS"]
#-----------------------------------------------------------------

# Calculate the confusion matrix of applying model on dataframe (including features and label) df with threshold
def calculate_confusion_matrix (model = None, df= None, threshold=None):
    model_prediction = [model]
    # Apply model on dataframe
    proba = Prediction_LGBM(trained_models=model_prediction, data = df, feature_names=name_features)
    proba_well = proba.loc[:, "model_0"]
    # Apply threshold
    if threshold==None: threshold = 0.5
    # Get label from dataframe df
    well_proba = proba_well.apply(lambda x: 1 if x >= threshold else 0)
    return well_proba

#------------------------------------------------------------------
# Load any compressed pickle file
file = "models/LightGBM_0.45.pbz2"
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data
# model_best = decompress_pickle(file)

# Loading Modeling
model_best = lgb.Booster(model_file="models/LGBM_20221125.json")

#-------------------------------------------------------------------------------------------------

col_1, col_2, col_3, col_4, col_5, = st.columns(5)
# with col_3:
    # st.image("https://i.ibb.co/Yd42K98/LogoVPI.png", width=250)
    # st.header("Welcome to Fracture Prediction Dashboard!")

#Loading data from browser:----------------------------------------
st.subheader("1. Load CSV input file:")
wells_df_predict = upload_csv()
st.write('---')
if wells_df_predict is not None:
    wells_df_predict = wells_df_predict.replace({-9999: np.nan}).dropna(how='any', subset = ["FRACTURE_ZONE"])

    
    st.write("Data Input:")
    st.dataframe(wells_df_predict.sort_index(), width=1400, height=300)
    st.write('---')
    st.write("Selected Prediction Model:")
    st.write(model_best)
#------------------------------------------------------------------
    feature_names = [col for col in wells_df_predict.columns if col not in ["WELL", "DEPTH","FRACTURE_ZONE"]]    
    # Full data for export data
    st.session_state.pred = st.button("Predict Fracture Zone")
    if st.session_state.pred:
        threshold = 0.23
        #Make label Prediction 
        predictions = (model_best.predict(wells_df_predict[feature_names])> threshold).astype(int)
        wells_df_predict['FRACTURE_ZONE_PRED'] = predictions
        st.dataframe(wells_df_predict, width=1400, height=300)
#Plot Data------------------------------------------------------------------
        plotting_curves = [c for c in wells_df_predict.columns.unique() if c not in ["DEPTH", "WELL", "TVD", "FRACTURE_ZONE", "FRACTURE_ZONE_PRED", "DCALI_FINAL", "INCL", "AZIM_TN"]]
        plotting_curves.sort()
        if "FRACTURE_ZONE_PRED" in wells_df_predict.columns.unique():
            plotting_curves.append("FRACTURE_ZONE_PRED")
        for well in wells_df_predict.WELL.unique():
            st.write('---')
            st.write(f"{well} Logs: \n")
            well_plot = wells_df_predict[wells_df_predict.WELL == well]
            charts_dict={}
            for i, c in enumerate(plotting_curves):
                charts_dict[i] = curve_plot(data=well_plot,filted_data=None, x_column=c)
    #Show Curve-----------------------------------------------------------------------
            st.write(alt.concat(*charts_dict.values()).configure(autosize='fit'))
    # Download --------------------------------------------------------------
        st.write('---')
        st.write("Download final result to csv file")

        st.download_button(label='Download All Wells',
                        data = wells_df_predict.to_csv(),
                        file_name='FracturePredictionALL.csv',
                        mime='text/csv')
        
