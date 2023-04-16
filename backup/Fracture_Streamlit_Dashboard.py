# import symbol
# from tracemalloc import DomainFilter
# from turtle import title
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
from Antuns.p2_plotting_data import selection_info, interval_define, make_selection, bar_plot, curve_plot

#Streamlit dashboard------------------------------------------------------------------------------------------
page_intro()
st.sidebar.success("Select a page!")

#Loading data from browser:-----------------------------------------------------------------------------------
st.subheader("1. Loading data:")
df = upload_csv()
if df is not None:
    st.dataframe(df.sort_index(), width=1400, height=300)

#Data Plotting ----------------------------------------------------------------------------------------------
    st.subheader("2. Plotting data:")
    st.session_state = selection_info(df)
    if st.session_state.method == "Single Well":
        df_single_well = df[df.well == st.session_state.option_w]
    else:
        df_single_well = df

#-----------------------------------------------------------------------
#Crossplot 2 curves
    interval = interval_define()

    col21, col22 = st.columns(2)
    with col21:
        selected_points = altair_component(make_selection(df_single_well, interval, st.session_state.option_x, st.session_state.option_y))
        if len(selected_points) > 0:
            del[selected_points['name']]
        # st.write(selected_points.items())
    
    with col22:
        #Histogram of curve in X Axis
        histogram_x = bar_plot(df_single_well, st.session_state.option_x)
        #Histogram of curve in Y Axis
        histogram_y =bar_plot(df_single_well, st.session_state.option_y)
        st.write(alt.hconcat(histogram_x,histogram_y))
        
        selected_df = None
        
        if len(selected_points) != 0:
            query = ' & '.join(
            f'{crange[0]} <= `{col}` <= {crange[1]}'
            for col, crange in selected_points.items())
            selected_df = df_single_well.query(query)
            st.write("Selected DataFrame")
            st.dataframe(selected_df, width=800, height=200,use_container_width=False)
        else:
            st.write("No Selection")
    #-----------------------------------------------------------------------
    # curves = df.columns
    # Store the initial value of widgets in session state
    # if "curves" not in st.session_state:
    #     st.session_state.curves = None

    # curves_selection = st.multiselect(
    #     "Curves",
    #     key="curves",
    #     options=curves.str.upper(),
    # )

    # if st.session_state.curves != None:
    #     num = len(st.session_state['curves'])
    #     # plotting_curves = st.session_state['curves']
    #     plotting_curves = ['gr', 'rhob', 'nphi', 'dts', 'dtc', 'lld','lls']
    #     charts_dict={}
    #     if plotting_curves != []:
    #         for i, c in enumerate(plotting_curves):
    #             charts_dict[i] = curve_plot(df_single_well, c)
                
    # st.write(ch for ch in charts_dict.values())

    #-----------------------------------------------------------------------
    plotting_curves = ['gr', 'rhob', 'nphi', 'dts', 'dtc', 'lld','lls']
    charts_dict={}
    if plotting_curves != []:
        for i, c in enumerate(plotting_curves):
            charts_dict[i] = curve_plot(data=df_single_well,filted_data=selected_df, x_column=c)

    #Show charts
    st.write(#(histogram_x&histogram_y)&
            alt.hconcat(charts_dict[0],charts_dict[1],charts_dict[2],charts_dict[3],
                        charts_dict[4],charts_dict[5],charts_dict[6]).configure(autosize='fit'))

    #-----------------------------------------------------------------------












#Download -------------------------------------------------------------------------------------------
# st.download_button(label='Download',
#                    data = df.query(query).to_csv(),
#                    file_name='Query_data.csv',
#                    mime='text/csv')
#-----------------------------------------------------------------------
