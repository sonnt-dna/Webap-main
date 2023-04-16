import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import altair as alt

import streamlit as st
import streamlit_nested_layout
from streamlit_vega_lite import altair_component

from PIL import Image

#PAGE SETTING------------------------------------------------------------------------------------------
img = Image.open("data/LogoVPI.png")
def page_intro():
    st.set_page_config(# Alternate names: setup_page, page, layout
                        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
                        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
                        page_title="VPI-MLogs",  # String or None. Strings get appended with "• Streamlit". 
                        page_icon=img,  # String, anything supported by st.image, or None.
    )
    col_1, col_2, col_3, col_4, col_5, = st.columns(5)
    with col_3:
        st.image("https://i.ibb.co/Yd42K98/LogoVPI.png", width=250)
    st.header("Welcome to VPI-MLOGS!")
    
    
#LOADING DATA------------------------------------------------------------------------------------------
def upload_csv():
    df = None
    uploaded_file = st.file_uploader(label='Upload *csv file from your drive! Choose a file:', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values=-9999)
        st.success("Loading finished!")
        st.dataframe(df, width=1400, height=300)
        st.write('---')
    return df

#PLOTTING------------------------------------------------------------------------------------------


#Selecte Functions----------------------------------------------------------------------
# Store the initial value of widgets in session state
def selection_info(df, method, option_w, option_x, option_y, option_c):
    if "method" not in st.session_state:
        st.session_state.method:str = "Single Well"
        st.session_state.option_w:str = "15-1-SNN-3P"
        st.session_state.option_x:str = "RHOB"
        st.session_state.option_y:str = "DTC"
        st.session_state.option_c:str = "WELL"
    well_names = np.sort(df.WELL.unique())
    st.radio("",
        key=method,
        options=["All Wells", "Single Well"],)
    st.radio(
        "WELL",
        key=option_w,
        options=well_names,)
    st.selectbox(
        "X Axis",
        key=option_x,
        options=(df.columns.sort_values().str.upper().drop(["WELL", "DEPTH"])),)
    st.selectbox(
        "Y Axis",
        key=option_y,
        options=(df.columns.sort_values().str.upper().drop(["WELL", "DEPTH"])),)
    st.selectbox(
        "Color Axis",
        key=option_c,
        options=df.columns.sort_values().str.upper())
    return st.session_state

#Interactive Charts-----------------------------------------------------------------------
@st.cache
def interval_define():
    return alt.selection_interval()

@st.cache
def make_selection(df, _interval, option_x, option_y, option_c):
    def c_(df, _interval, option_x, option_y, x_log:str="linear", y_log:str="linear"):
        return alt.Chart(df,
                        title="Crossplot "+option_x+" vs "+option_y+"",
                        ).mark_point().encode(
                        x = alt.X(option_x.upper(),
                                axis=alt.Axis(title=option_x),
                                scale= alt.Scale(zero=False, type=x_log
                                    )
                                ),
                        y = alt.Y(option_y.upper(),
                                axis=alt.Axis(title=option_y),
                                scale=alt.Scale(zero=False,type=y_log
                                )
                                ),
                        color=alt.condition(_interval, option_c, alt.value('lightgray')),
                        ).properties(
                        selection=_interval,
                        height=570,
                        width=600)#.transform_regression(option_x.upper(), option_y.upper()).mark_line()
                        
    if option_x in ["LLD", "LLS"]:
        x_log = "log"
    else:
        x_log = "linear"
        
    if option_y in ["LLD", "LLS"]:
        y_log = "log"
    else:
        y_log = "linear"
    return c_(df, _interval, option_x, option_y, x_log, y_log)

#Histogram-----------------------------------------------------------------------
def bar_plot(data, option_x):
    def c_(data, option_x, _log):
        return alt.Chart(title="Histogram of "+option_x+"",
                        data=data
                        ).mark_bar().encode(
                                            x = alt.X(option_x.upper(),
                                                    bin=alt.Bin(maxbins=30),
                                                    axis=alt.Axis(title=option_x),
                                                    scale=alt.Scale(zero=False)
                                                    ),
                                            y = alt.Y('count()',
                                                    axis=alt.Axis(title='Number of Values'),
                                                    scale=alt.Scale(zero=False, type=_log),
                                                    ),
                                            color = alt.Color('WELL', legend=None
                                                            )
                                            ).properties(
                                                height=250,
                                                width=250
                                            )
    if option_x in ["LLD", "LLS"]:
        return c_(data, option_x, "symlog")
    else:
        return c_(data, option_x, "linear")
        
#Curve View-----------------------------------------------------------------------
def curve_plot(data,filted_data, x_column):
    def c_(data,filted_data, x_column, _log):
        color_codes = {"GR":"lime",
                        "LLD":"red",
                        "LLS":"dodgerblue",
                        "NPHI":"blue",
                        "RHOB":"red",
                        "DTC":"red",
                        "DTS":"magenta",
                        "FRACTURE_ZONE":"lightcoral",
                        "FRACTURE_ZONE_PRED":"lightgreen"
                        }
        if x_column in color_codes.keys():
            color_ = color_codes[x_column]
        else:
            color_ = "blue"
        return alt.Chart(data
                        ).mark_line(size=1,
                                    orient='horizontal',
                                    color=color_,
                                    point=alt.OverlayMarkDef(color="", size=1) #Show raw points
                                    ).encode(
                                            x=alt.X(x_column.upper(),
                                                    scale=alt.Scale(zero=False, type=_log),
                                                    axis=alt.Axis(title=x_column.upper(),
                                                                titleAnchor='middle',
                                                                orient='top',
                                                                labelAngle=0,
                                                                titleColor=color_,
                                                                labelColor=color_,
                                                                tickColor=color_,
                                                                )
                                                    ),
                                            y=alt.Y('DEPTH', 
                                                    scale=alt.Scale(zero=False,
                                                                    reverse=True,
                                                                    ),
                                                    axis=alt.Axis(title=None,
                                                                labelColor=color_,
                                                                tickColor=color_,
                                                                  )
                                                    )
                                    ).properties(height=700,
                                                width=129
                                    )
                                    
    if x_column in ["LLD", "LLS"]:
        curve = c_(data,filted_data, x_column, "log")
    else:
        curve = c_(data,filted_data, x_column, "linear")
        
    if filted_data is not None: 
        point_plot = alt.Chart(filted_data).mark_circle(size=20,
                                                        color='red',
                                                        opacity=1
                                                        ).encode(
                                                        x=x_column,
                                                        y='DEPTH'
                                                        )           
        return curve + point_plot
    else:
        return curve
#MissingBar-----------------------------------------------------------------------
def missing_bar(data, x_title):
    return alt.Chart(data).mark_bar().encode(
            x=alt.X('Columns', sort='-y', title=x_title),
            y='Count missing (%)',
            color=alt.condition(
                alt.datum['Count missing (%)'] >10,  # If count missing is > 10%, returns True,
                alt.value('orange'),             # which sets the bar orange.
                alt.value('steelblue')           # And if it's not true it sets the bar steelblue.
            )
            ).properties(
            width=500,
            height=250
            ).configure_axis(
            grid=False
            )
#BoxPLot-----------------------------------------------------------------------
def missing_box(data, curve):
    if curve in ["LLD", "LLS"]:
        return alt.Chart(data).mark_boxplot(extent='min-max').encode(
            x=alt.X('WELL:O', title=None, 
                    ),
            y=alt.Y(f'{curve}:Q', title=curve,scale=alt.Scale(zero=False, type="log")
                    ),
            color='WELL:N'
        ).properties(
            width=500,
            height=300
        )
    else:
        return alt.Chart(data).mark_boxplot(extent='min-max').encode(
            x=alt.X('WELL:O', title=None
                    ),
            y=alt.Y(f'{curve}:Q', title=curve,scale=alt.Scale(zero=False)
                    ),
            color='WELL:N'
        ).properties(
            width=500,
            height=300
        )
#Histogram Line-----------------------------------------------------------------------
def hist_line_plot(data, curve):
    st.caption(f"Histogram of {curve}")
    if curve in ["LLD", "LLS"]:
        fig = sns.displot(data, x=curve, hue="WELL", kind="kde", height=5,aspect=1.2, log_scale=True)
        fig.set(ylabel="Values")
        st.pyplot(fig)
    else:
        fig = sns.displot(data, x=curve, hue="WELL", kind="kde", height=5,aspect=1.2)
        fig.set(ylabel="Values")
        st.pyplot(fig)
#CrossPlot-----------------------------------------------------------------------
def crossplot(data, x_curve, y_curve):     
    fig = sns.jointplot(data=data, x=x_curve, y=y_curve, hue="WELL")
    if x_curve in ["LLD", "LLS"]:
        fig.ax_joint.set_xscale('log')
        fig.ax_marg_x.set_xscale('log')
    if y_curve in ["LLD", "LLS"]:
        fig.ax_joint.set_yscale('log')
        fig.ax_marg_y.set_yscale('log')
    st.pyplot(fig)
#PairPlot-----------------------------------------------------------------------
def pairplot(data, rows, cols,color_):
    return alt.Chart(data).mark_circle().encode(
            alt.X(alt.repeat("column"), type='quantitative', scale=alt.Scale(zero=False)),
            alt.Y(alt.repeat("row"), type='quantitative', scale=alt.Scale(zero=False)),
            color=color_
            ).properties(
                width=100,
                height=100
            ).repeat(
                row = rows,
                column = cols
            ).configure_axis(
                grid=False
            )
#Heatmap----------------------------------------------------------------
def heatmap(df):            
    fig = sns.heatmap(df, annot=True)
    st.pyplot(fig)
#Heatmap----------------------------------------------------------------
def plotly_3d(data, x, y, z, color, size, symbol, log_x, log_y, log_z):
    #Data slicer
    curvs_ = curves_list(data, no_well=True)
    def slicer_(data, sli_key, val_key,):
        slicer1_, slicer2_ = st.columns([1,7])
        # sli=curvs_[0]
        with slicer1_:
            sli = st.selectbox("Data slicer", key=sli_key, options=curvs_)
        with slicer2_:
            values = st.slider('Select a range of values', 
                                min_value = float(data[sli].min()), 
                                max_value = float(data[sli].max()), 
                                value=(float(data[sli].min()), float(data[sli].max())), 
                                key=val_key,
                                )
        data = data.query(f"{sli} >= {values[0]} and {sli} <= {values[1]}")
        return data
    data = slicer_(data, "slicer_1", "sli1_value")
    data = slicer_(data, "slicer_2", "sli2_value")
    data = slicer_(data, "slicer_3", "sli3_value")
         
    fig = px.scatter_3d(data, x=x, 
                        y=y, 
                        z=z,
                        color=color, 
                        size=size, 
                        size_max=18,
                        symbol=symbol, 
                        opacity=0.7,
                        log_x=log_x,
                        log_y=log_y,
                        log_z = log_z,
                        width=1000, height=700,
                        color_continuous_scale="blugrn")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), #tight layout
                    #   paper_bgcolor="LightSteelBlue"
                    template="none") 
    st.plotly_chart(fig)
#Curves_list----------------------------------------------------------------
def curves_list(data:pd.DataFrame, no_depth:bool=None, no_well:bool=None):
    curs = list(data.columns.unique())
    curs.sort()
    if no_depth == True:
        curs.remove("DEPTH")
    if no_well == True:
        curs.remove("WELL")
    return curs
#----------------------------------------------------------------
    