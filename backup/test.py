from turtle import width
import pandas as pd
import numpy as np

import streamlit as st

import matplotlib.pyplot as plt
import altair as alt
from streamlit_vega_lite import altair_component

from io import StringIO

from Antuns.page_setting import page_intro
from Antuns.p1_import_csv import upload_csv
from Antuns.p2_plotting_data import interval_define, make_selection, bar_plot, curve_plot


df = pd.read_csv("/Users/vpi103/Desktop/DnA/AI_Fractures/EDA_dataprep/Git-Fracture_WebApp/G3_Data.csv",na_values=-9999)
df = df.drop(df.columns[0], axis=1)
df = df.rename(str.lower, axis='columns')
df = df.rename(columns={'fracture_zone':'fracturezone', 'fracture intensity':'fractureintensity'})
df = df.sample(5000, random_state=9)
df = df[df.well == "15-1-SNN-3P"]
df = df.reset_index()
st.dataframe(df.sort_index(), width=1400, height=300)
st.write(len(df))

def reset_session():
    for key in st.session_state.keys():
        st.write(st.session_state[key])
        
# st.write(st.session_state)
st.write('---')
#-----------------------------------------------------------------------
df_filter = df.copy()
# st.dataframe(df_filter.loc[[2170],:])
st.write('---')
x = 0
while x < 10:
    
    st.write(st.session_state)
    st.write('---')
    
#-----------------------------------------------------------------------
    # st.dataframe(df_filter.loc[[2170],:])
    #Crossplot 2 curves
    interval = interval_define()
    col21, col22 = st.columns(2)
    with col21:
        selected_points = altair_component(make_selection(df_filter, interval, "gr","rhob"), key=str(x*x))
        if len(selected_points) > 0:
            del[selected_points['name']]

    with col22:
        selected_df = None
        if len(selected_points) != 0:
            query = ' & '.join(
            f'{crange[0]} <= `{col}` <= {crange[1]}'
            for col, crange in selected_points.items())
            selected_df = df_filter.query(query)
            st.write(f"Total selected points: {len(selected_df)}")
            st.dataframe(selected_df, width=800, height=260,use_container_width=False)
        else:
            st.write("No Selection")
    #-----------------------------------------------------------------------
    st.write('---')
    st.write("Outliers Removal:")
    # df_nomarlized = df.copy()
    col1, col2 = st.columns([3,3])
    with col1:
        curve_editting = st.selectbox("Select curve to edit:",
                                    key="selected_curv",
                                    options=(df_filter.columns.sort_values().str.upper()),
                                    )
    with col2:
        n_value = int(st.text_input("Enter number of rows", "5"))
        st.write("Number of rows: ", n_value)

    def normalize_outlier(df_nomarlized, selected_df, curve, n_value):
        n=n_value//2
        for i in selected_df.index:
            df_nomarlized.loc[[i],curve.lower()] = df_nomarlized.loc[i-n:i+n,curve.lower()].mean()
        return df_nomarlized

    # st.write(st.session_state)

    if st.button("Remove Outliers"):
        df_filter = normalize_outlier(df_filter, selected_df, curve_editting, n_value)
        selected_df = None
        # st.dataframe(df_filter.loc[[2170],:])
        x +=1
    #-----------------------------------------------------------------------
    st.write('---')
    st.write("Curves view:")
    plotting_curves = ['gr', 'rhob', 'nphi', 'dts', 'dtc', 'lld','lls']
    charts_dict={}
    if plotting_curves != []:
        for i, c in enumerate(plotting_curves):
            charts_dict[i] = curve_plot(data=df_filter,filted_data=selected_df, x_column=c)
    #-----------------------------------------------------------------------
    #Show charts
    st.write(
            alt.hconcat(charts_dict[0],charts_dict[1],charts_dict[2],charts_dict[3],
                            charts_dict[4],charts_dict[5],charts_dict[6]).configure(autosize='fit'))
