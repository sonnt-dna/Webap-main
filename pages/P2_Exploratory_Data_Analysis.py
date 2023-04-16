import pandas as pd
import numpy as np
import os

import streamlit as st
import streamlit_nested_layout
from streamlit_vega_lite import altair_component

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import altair as alt

import importlib

import fx
importlib.reload(fx)

#Streamlit dashboard------------------------------------------------------------------------------------------
fx.page_intro()

#Loading data from browser:-----------------------------------------------------------------------------------
st.subheader("1. Data Loading:")
df = fx.upload_csv()

#-----------------------------------------------------------------------
if df is not None:
    df = df.rename(str.upper, axis='columns')
    if "DCALI_FINAL" in df.columns.unique():
        df = df.rename(columns={'DCALI_FINAL':'DCAL'})
    if "FRACTURE_ZONE" in df.columns.unique():
        df = df.rename(columns={'FRACTURE_ZONE':'FRACTUREZONE'})
    if "FRACTURE INTENSITY" in df.columns.unique():
        df = df.rename(columns={'FRACTURE INTENSITY':'FRACTURE_INTENSITY'})
    # df_raw = df.dropna(subset=['fracturezone'], how='any')
    # if len(df) > 5001:
        # df = df.sample(5000, random_state=9) #Resample dataframe to 5000 rows
    else:
        pass
    if "FRACTUREZONE" in df.columns.unique():
        df = df.reset_index().drop(["index", "FRACTUREZONE"], axis=1)
    # if "FRACTURE_INTENSITY" in df.columns.unique():
    #     df = df.reset_index().drop(["index", "FRACTURE_INTENSITY"], axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
#-----------------------------------------------------------------------
st.write('---')
if df is not None:
    curves = fx.curves_list(df)
    curves.remove("DEPTH"); curves.remove("WELL")
    st.dataframe(df.sort_index(), width=1400, height=300)

#Data Plotting ----------------------------------------------------------------------------------------------
    st.write('---')
    st.subheader("2. Exploratory Data Analysis:")
#-----------------------------------------------------------------------
    col1, col2 = st.columns([1,8])
    well_names = np.sort(df.WELL.unique())
    
    if "method_eda" not in st.session_state:
        st.session_state.method_eda:str = "Single Well"
        st.session_state.well_eda:str = well_names[0]
        
    with col1:
        st.radio("",
                key="method_eda",
                options=["All Wells", "Single Well"])
        st.radio("WELL",
                key="well_eda",
                options=well_names)
#-----------------------------------------------------------------------
    def well_filter(well_name):
        return df[df.WELL == well_name]
    with col2:
        st.write('Data Description:')
        if st.session_state.method_eda == "All Wells":
            st.dataframe(df.describe(),width=1400, height=300)
        elif st.session_state.method_eda == "Single Well":
            df_single_w = df[df.WELL == st.session_state.well_eda]
            st.dataframe(df_single_w.describe(),width=1400, height=300)
        else:
            pass
        #Action buttons----------------------------------------------------------------
    st.write('---')
    eda1, eda2 = st.columns([1,8])
    with eda1:
        st.radio("Data Exploratory",
                 key="eda_plt",
                 options=["Missing Statistic", "Curve Distribution", 
                          "Histogram Overlay", "Crossplot", 
                          "Pair Plot", "3D Scatter"]
                 )
        #Missing stats----------------------------------------------------------------
    with eda2:
        if st.session_state.eda_plt == "Missing Statistic":
            st.write('Missing Data Statistic:')
            def missing_count(df):
                missing = df.isnull().sum()*100/df.isnull().sum().sum()
                missing = missing[missing >= 0].reset_index()
                missing.columns = ['Columns', 'Count missing (%)']
                return missing
            mt1, mt2, mt3 = st.columns(3)
            with mt1:
                st.caption("Missing data rate of whole wells")
                st.write(fx.missing_bar(missing_count(df), "ALL WELLS"))
                for i, w in enumerate(df.WELL.unique()):
                    if i%3 == 0:
                        st.caption(f"Missing data rate of {w}")
                        # st.dataframe(well_filter(w))
                        st.write(fx.missing_bar(missing_count(well_filter(w)), f"WELL {w}"))
            with mt2:
                for i, w in enumerate(df.WELL.unique()):
                    if i%3 == 1:
                        st.caption(f"Missing data rate of {w}")
                        st.write(fx.missing_bar(missing_count(well_filter(w)), f"WELL {w}"))
            with mt3:
                for i, w in enumerate(df.WELL.unique()):
                    if i%3 == 2:
                        st.caption(f"Missing data rate of {w}")
                        st.write(fx.missing_bar(missing_count(well_filter(w)), f"WELL {w}"))
        #Missing Box----------------------------------------------------------------
        if st.session_state.eda_plt == "Curve Distribution":
            st.caption("Curve Distribution")
            mb1, mb2, mb3 = st.columns(3)
            for i, c in enumerate(curves):
                if i%3 == 0:
                    with mb1:
                        st.caption(f"Distribution of {c}")
                        st.write(fx.missing_box(df, c))
                if i%3 == 1:
                    with mb2:
                        st.caption(f"Distribution of {c}")
                        st.write(fx.missing_box(df, c))
                if i%3 == 2:
                    with mb3:
                        st.caption(f"Distribution of {c}")
                        st.write(fx.missing_box(df, c))
        #Histogram Line----------------------------------------------------------------
        if st.session_state.eda_plt == "Histogram Overlay":
            h1, h2, h3 = st.columns(3)
            for i, c in enumerate(curves):
                if i%3 == 0:
                    with h1:
                        fx.hist_line_plot(df,c)
                if i%3 == 1:
                    with h2:
                        fx.hist_line_plot(df,c)
                if i%3 == 2:
                    with h3:
                        fx.hist_line_plot(df,c)
        #CrossPlot---------------------------------------------------------------- 
        if st.session_state.eda_plt == "Crossplot":
            pair_curv = [(a, b) for idx, a in enumerate(curves) for b in curves[idx + 1:]]
            cp0, cp1, cp2, cp3, cp4 = st.columns(5)
            for i, c in enumerate(pair_curv):
                if i%5 == 0:
                    with cp0:
                        fx.crossplot(df, pair_curv[i][0], pair_curv[i][1])
                if i%5 == 1:
                    with cp1:
                        fx.crossplot(df, pair_curv[i][0], pair_curv[i][1])
                if i%5 == 2:
                    with cp2:
                        fx.crossplot(df, pair_curv[i][0], pair_curv[i][1])
                if i%5 == 3:
                    with cp3:
                        fx.crossplot(df, pair_curv[i][0], pair_curv[i][1])
                if i%5 == 4:
                    with cp4:
                        fx.crossplot(df, pair_curv[i][0], pair_curv[i][1])
        #Pairpot----------------------------------------------------------------
        if st.session_state.eda_plt == "Pair Plot":
            p1_, p2_ = st.columns([1,7])
            if "pair_opt" not in st.session_state:
                st.session_state.pair_opt:str = "ALL WELLS"
                st.session_state.color_pair:str = "WELL"
                st.session_state.well_pair:str = list(df.WELL.unique())[0]
            with p1_:
                pair_opt_ = st.radio("", key="pair_opt", options=["ALL WELLS", "SINGLE WELL"])
                well_pair_ = st.selectbox("WELL", key="well_pair", options=list(df.WELL.unique()))
                colorp_ = st.selectbox("COLOR", key="color_pair", options=fx.curves_list(df))
            if pair_opt_ == "ALL WELLS":
                st.write(fx.pairplot(df, curves, curves, colorp_))
            elif pair_opt_ == "SINGLE WELL":
                st.write(fx.pairplot(df[df["WELL"]==well_pair_], curves, curves, colorp_))
            else:
                st.write("Undefined Error!")
                
        #3D Plotly----------------------------------------------------------------
        if st.session_state.eda_plt == "3D Scatter":
            
            wells_ = list(df.WELL.unique())
            curvs_ = fx.curves_list(df, no_well=True)
            colors_ = fx.curves_list(df)
            sizes_ = ["WELL", "FRACTURE_INTENSITY", "DEPTH", None]
            symbols_ = ["WELL", "FRACTURE_INTENSITY", None]
            
            if "well_3d" not in st.session_state:
                st.session_state.w_opt:str = "ALL WELLS"
                st.session_state.well_3d:str = wells_[0]
                st.session_state.x_3d:str = curvs_[0]
                st.session_state.y_3d:str = curvs_[0]
                st.session_state.z_3d:str = curvs_[0]
                st.session_state.color_3d:str = "WELL"
                st.session_state.size_3d:str = "DEPTH"
                st.session_state.symbol_3d:str = "WELL"
                
            p1_, p2_ = st.columns([1,7])
            with p1_:
                w_opt = st.radio("", key="w_opt", options=["ALL WELLS", "SINGLE WELL"])
                well_ = st.selectbox("WELL", key="well_3d", options=wells_)
                x_ = st.selectbox("X", key="x_3d", options=curvs_)
                y_ = st.selectbox("Y", key="y_3d", options=curvs_)
                z_ = st.selectbox("Z", key="z_3d", options=curvs_)
                color_ = st.selectbox("COLOR", key="color_3d", options=colors_)
                size_ = st.selectbox("SIZE", key="size_3d", options=sizes_)
                symbol_ = st.selectbox("SYMBOL", key="symbol_3d", options=symbols_)
            with p2_:
                log_x, log_y, log_z = [False, False, False]
                if x_ in ["LLD", "LLS"]:
                    log_x = True
                if y_ in ["LLD", "LLS"]:
                    log_y = True
                if z_ in ["LLD", "LLS"]:
                    log_z = True
                if w_opt == "ALL WELLS":
                    fx.plotly_3d(df, x_, y_, z_, color_, size_, symbol_, log_x, log_y, log_z)
                else:
                    df_3d_plt = df[df["WELL"]==well_]
                    fx.plotly_3d(df_3d_plt, x_, y_, z_, color_, size_, symbol_, log_x, log_y, log_z)
                    
#Data Plotting ----------------------------------------------------------------------------------------------
    st.write('---')
    st.subheader("3. Outliers Processing:")
    _o1, _o2 = st.columns([1,8])
    with _o1:
        st.session_state = fx.selection_info(df,"method", "option_w", "option_x", "option_y", "option_c")

            
#Crossplot and bar plot-----------------------------------------------------------------------
    with _o2:
        def rm_outliers(data):
            interval = fx.interval_define()
            col21, col22 = st.columns(2)
            with col21:
                selected_points = altair_component(fx.make_selection(data, 
                                                                interval, 
                                                                st.session_state.option_x, 
                                                                st.session_state.option_y, 
                                                                st.session_state.option_c,
                                                                )
                                                )
                if len(selected_points) > 0:
                    del[selected_points['name']]
            
            with col22:
                selected_df = None
                if len(selected_points) != 0:
                    query = ' & '.join(
                    f'{crange[0]} <= `{col}` <= {crange[1]}'
                    for col, crange in selected_points.items())
                    selected_df = data.query(query)
                    st.write(f"Total selected points: {len(selected_df)}")
                    st.dataframe(selected_df, width=800, height=260,use_container_width=False)
                else:
                    st.write("No Selection")
                
                if selected_df is not None:
                    st.write("Histogram of selected data:")
                    histogram_x = fx.bar_plot(selected_df, st.session_state.option_x)
                    histogram_y = fx.bar_plot(selected_df, st.session_state.option_y)
                    st.write(alt.hconcat(histogram_x,histogram_y))
                else:
                    st.write("Histogram of entire data:")
                    histogram_x = fx.bar_plot(data, st.session_state.option_x)
                    histogram_y = fx.bar_plot(data, st.session_state.option_y)
                    st.write(alt.hconcat(histogram_x,histogram_y))

        #Outlier Removal-----------------------------------------------------------------------
            st.write('---')
            df_nomarlized = data.copy()
            col1, col2 = st.columns([1,8])
            with col1:
                plotting_curves = st.multiselect("Select curves to plot:", key="curv_plt", options=fx.curves_list(data, no_depth=True, no_well=True))
                curve_editting = st.selectbox("Select curve to edit:",
                                            key="selected_curv",
                                            options=plotting_curves,
                                            )
                n_value = int(st.text_input("Number of rows for Mean calculation ", "5"))

                def normalize_outlier(df_nomarlized, selected_df, curve, n_value):
                    n=n_value//2
                    for i in selected_df.index:
                        df_nomarlized.loc[[i],curve.upper()] = df_nomarlized.loc[i-n:i+n,curve.upper()].mean()
                    return df_nomarlized
                def remove_data_point(df_nomarlized, selected_df, curve):
                    for i in selected_df.index:
                        # df_nomarlized[i, curve] = 0                   #ERROR ALARM!!!!
                        df_nomarlized = df_nomarlized.drop(index=i)     #ERROR ALARM!!!!
                    return df_nomarlized
                    
                if st.button("Outliers Processing"):
                    st.session_state.fdata = normalize_outlier(df_nomarlized, selected_df, curve_editting, n_value)
                    _well = "".join((st.session_state.fdata.WELL.unique()).tolist())
                    st.session_state.loc_data = pd.concat([df[(df["WELL"] != _well)],st.session_state.fdata], axis=0)
                    selected_df = None
                if st.button("Remove"):
                    st.session_state.fdata = remove_data_point(df_nomarlized, selected_df, curve_editting)
                    _well = "".join((st.session_state.fdata.WELL.unique()).tolist())
                    st.write(_well)
                    st.write(type(_well))
                    st.session_state.loc_data = pd.concat([df[(df["WELL"] != _well)],st.session_state.fdata], axis=0)
                    selected_df = None

        #Curve View-----------------------------------------------------------------------
            with col2:
                charts_dict={}
                def plt_curs(data, option_w):
                    data_plt = data[data["WELL"] == option_w]
                    if plotting_curves != []:
                        for i, c in enumerate(plotting_curves):
                            charts_dict[i] = fx.curve_plot(data=data_plt,filted_data=selected_df, x_column=c)
                if 'loc_data' not in st.session_state:
                    plt_curs(df_nomarlized, st.session_state.option_w)
                else:
                    plt_curs(st.session_state.loc_data, st.session_state.option_w)
                    
        #Show Curve-----------------------------------------------------------------------
                st.write(alt.concat(*charts_dict.values()).configure(autosize='fit'))#.configure_concat(spacing=0))
                
        #------------------------
        def check_method(df):
            if st.session_state.method == "Single Well":
                data = df[df.WELL == st.session_state.option_w]
                data = data.sort_values(by=['DEPTH'])
                data = data.reset_index().drop(["index"], axis=1)
            else:
                data = df
            return data
        #------------------------
        
        if 'loc_data' not in st.session_state:
            data = check_method(df)
        else:
            data = check_method(st.session_state.loc_data)       
        
        rm_outliers(data)
            
# Download --------------------------------------------------------------
    st.write('---')
    st.write("Download final result to csv file")
    if "loc_data" not in st.session_state:
        saving_df = df
    else:
        saving_df = st.session_state.loc_data
    st.download_button(label='Download',
                    data = saving_df.to_csv(),
                    file_name='Query_data.csv',
                    mime='text/csv')

