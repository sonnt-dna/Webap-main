import pandas as pd
import matplotlib.pyplot as plt
import lasio
import numpy as np
import io
import streamlit as st

import altair as alt
from streamlit_vega_lite import altair_component

from io import StringIO

from Antuns.page_setting import page_intro
from Antuns.p1_import_csv import upload_csv
from Antuns.p2_plotting_data import selection_info, interval_define, make_selection, bar_plot, curve_plot

#-----------------------------------------------------------------------------
page_intro()

#1_LOADINGDATA-----------------------------------------------------------------
st.write("Log ASCII standard (LAS) is a standard file format common in the oil-and-gas and water well industries to store well log information. Well logging is used to investigate and characterize the subsurface stratigraphy in a well.")
st.subheader('To begin using the app, load your LAS file using the file upload option below.')
st.subheader("1. LAS File Loading:")

@st.cache_data() #allow_output_mutation=True, suppress_st_warning=True
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            str_io = StringIO(bytes_data.decode('Windows-1252'))
            las = lasio.read(str_io)
            well_data = las.df()
            well_data['DEPTH'] = well_data.index

        except UnicodeDecodeError as e:
            st.error(f"error loading log.las: {e}")
    else:
        las = None
        well_data = None

    return las, well_data

#TODO
def missing_data():
    st.title('Missing Data')
    missing_data = well_data.copy()
    missing = px.area(well_data, x='DEPTH', y='DT')
    st.plotly_chart(missing)

# Sidebar Options & File Upload
las=None

uploadedfile = st.file_uploader(' ', type=['.las'])
las, well_data = load_data(uploadedfile)
# print("In las 01 file:", las)
# print("In las 01 file:", well_data)
if las:
    st.success('File Uploaded Successfully')
    st.write(f'<b>Well Name</b>: {las.well.WELL.value}',unsafe_allow_html=True)


#2_CURVES_INFOMATION-----------------------------------------------------------------

    st.subheader("2. Curve logs details:")

    st.caption("All curve logs in data:")
    curves = []
    for curve in las.curves:
        
        st.write(curve.mnemonic)
        curves.append(curve.mnemonic)
        
    for count, curve in enumerate(las.curves):
        st.write("---")
        st.write(f"Curve: {curve.mnemonic}, \t Units: {curve.unit}, \t Description: {curve.descr}")
        st.write(f"There are a total of: {count+1} curves present within this file")

#3_DATAFRAME-----------------------------------------------------------------

    st.subheader("3. Converting LAS to DataFrame:")
    st.caption("Preview of all Dataframe")
    well = las.df()
    st.write(well.head())
    st.caption("Well curves Statistics")
    st.write(well.describe())

#4_DOWNLOAD-----------------------------------------------------------------
    st.subheader("4. Data Preprocessing:")
    st.session_state.old_name:str
    st.session_state.new_name:str
    st.session_state.changename:bool
    st.session_state.well = None

        
    st.write("4.1 Rename curves")
    col1, col2, col3 = st.columns([3,3,1])
    with col1:
        st.session_state.old_name = st.selectbox("Select curve to rename",curves)
    with col2:
        st.session_state.new_name = st.text_input("New curve name")
    with col3:
        st.session_state.changename = st.button("Change")
    if st.session_state.changename:
        st.session_state.well = well.rename(columns={st.session_state.old_name:st.session_state.new_name})
        
    
    # st.write(st.session_state)
    st.dataframe(st.session_state.well,width=1400)
    

#5_DOWNLOAD-----------------------------------------------------------------

    st.subheader("5. Download CSV file:")
    st.download_button(label='Download CSV File',
                data = well.to_csv(),
                file_name=f"{las.well.WELL.value}.csv",
                mime='text/csv')

st.subheader("2. Multiple LAS Files Loading:")

# Define function to load LAS file data into DataFrame
@st.cache_data() #allow_output_mutation=True, suppress_st_warning=True
def load_data(uploaded_files):
    dataframes = {}
    las_data_list = []
    las_data = []
    if uploaded_files is not None:
        for file in uploaded_files:
            try:
                bytes_data = file.read()
                str_io = StringIO(bytes_data.decode('Windows-1252'))
                las_data = lasio.read(str_io)
                # print("in las:", las_data)
                well_data = las_data.df()
                # print("in las:", well_data)
                well_data['WELL'] = las_data.well.WELL.value
                # well_data['DEPTH'] = well_data.index
                if well_data.index.name == 'DEPT':
                    well_data.reset_index('DEPT', inplace=True)  # Set 'DEPT' as the new index
                    well_data.index.name = 'DEPT' 
                    if len(well_data) > 0:  # Kiểm tra xem dataframe có dữ liệu không
                        dataframes[file.name] = well_data
                        las_data_list.append(las_data)
                    else:
                        st.warning(f"No data in file {file.name}")
                else: 
                    well_data.index.name == 'DEPTH'
                    well_data.reset_index('DEPTH', inplace=True)  # Set 'DEPTH' as the new index
                    well_data.index.name = 'DEPTH' 
                    if len(well_data) > 0:  # Kiểm tra xem dataframe có dữ liệu không
                        dataframes[file.name] = well_data
                        las_data_list.append(las_data)                   
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
    return dataframes, las_data_list, las_data
            # if uploaded_files is not None:
                # try:
                #     bytes_data = uploaded_files.read()
                #     str_io = StringIO(bytes_data.decode('Windows-1252'))
                #     las_file = lasio.read(str_io)
                #     well_data = las_file.df()
                #     well_data['DEPTH'] = well_data.index

                # except UnicodeDecodeError as e:
                #     st.error(f"error loading log.las: {e}")
                # else:
                #     las = None
                #     well_data = None

                # return las, well_data

# Sidebar Options & File Upload
uploaded_files = st.file_uploader(label='Upload LAS files:', accept_multiple_files=True, type='las')

dataframes, las_data_list, las_data = load_data(uploaded_files)
# print("print las data", las_data_list)
well_names = {}
if dataframes:
    merged_df = []
    for file_name, df in dataframes.items():
        well_name = file_name.split(".")[0]
        # st.write(f"Data for {well_name}:")
        # Lấy danh sách các tên giếng
        well_names = list(dataframes.keys())

    # Cho phép người dùng chọn giếng và hiển thị DataFrame tương ứng
    selected_well = st.selectbox("Select Well", well_names, key = "selected_well_1")

    # st.write(f"Data for {selected_well}:")
    st.write(dataframes[selected_well])
    # Tạo một danh sách các DataFrame
    dfs = [df for _, df in dataframes.items()]
    merged_df = pd.concat([df for df in dfs])
    # Hiển thị DataFrame tổng thể
    st.write("Merged DataFrame:")
    st.write(merged_df)
else:
    st.warning('No valid LAS files were uploaded.')

curves = []
wellname = []
las = las_data

if las:
    
    list_well = []
    for las_data in las_data_list:
        well_name = las_data.well['WELL'].value
        list_well.append(well_name)
        # print(list_well)

    st.success('File Uploaded Successfully')
    st.write(f'<b>Well Name</b>: {list_well}', unsafe_allow_html=True)
    # 2_CURVES_INFOMATION-----------------------------------------------------------------
    if las:
        
        st.subheader("2. Curve logs details:")
        selected_well = st.selectbox("Select Well", well_names, key = "selected_well_2")
        st.caption("All curve logs in data:")
        
        curves = []

        for well, las_file in zip(well_names, las_data_list):
            if well == selected_well:
                las = las_file
                break

    
        # print("in ra las:", las)
        for curve in las.curves:
            st.write(curve.mnemonic)
            curves.append(curve.mnemonic)

        for count, curve in enumerate(las.curves):
            st.write("---")
            st.write(f"Curve: {curve.mnemonic}, \t Units: {curve.unit}, \t Description: {curve.descr}")
            st.write(f"There are a total of: {count+1} curves present within this file")

    #3_DATAFRAME-----------------------------------------------------------------
        if "selected_well" not in st.session_state:
            st.session_state.selected_well = None
            st.session_state.selected_well_multi = None
        st.subheader("3. Converting LAS to DataFrame:")
        st.caption("3.1 Preview of all Dataframe")

        selected_well = st.selectbox("Select Well", well_names, key = "selected_well_5")
        # print("Well_name", well_names)
        # print("las_data_list", las_data_list)
        for well, las_file in zip(well_names, las_data_list):
            if well == selected_well:
                las = las_file
                break
            # break
        well = las.df()
        well['WELL'] = las.well.WELL.value
        well['DEPTH'] = well.index
        well = well.reset_index(drop=True)
        well = well.reindex(columns=['DEPTH'] + [col for col in well.columns if col != 'DEPTH'])
        st.write(well.head())
        st.caption("3.2 Well curves Statistics")
        st.write(well.describe())
        # print("in ra danh sách giếng", list_well)
        # create a selectbox to choose the well
    
    selected_well_multi = st.multiselect(" 3.3 Select well for download", list_well)
    st.session_state.changename = st.button("Create", key="create_curve")
    if st.session_state.changename:
        dataframes_df = pd.concat(dataframes.values(), ignore_index=True)
        st.session_state.selected_well_multi = dataframes_df.loc[dataframes_df['WELL'].isin(selected_well_multi)].reset_index(drop=True)
        st.dataframe(st.session_state.selected_well_multi)
        st.write(" Download DataFrame")
        st.download_button(label='Download CSV File',
                    data = st.session_state.selected_well_multi.to_csv(),
                    file_name=f"{selected_well_multi}.csv",
                    mime='text/csv')

    #4_Data Preprocessing-----------------------------------------------------------------
    st.subheader("4. Data Preprocessing:")
    st.session_state.old_name:str
    st.session_state.new_name:str
    st.session_state.changename:bool
    st.session_state.well = None

        
    st.write("4.1 Rename curves")
    # Hiển thị danh sách các giếng trong một ô chọn giếng
    # selected_well_rename = st.selectbox("Select Well", list_well)
    # col1, col2, col3 = st.columns([3,3,1])
    # with col1:
    #     st.session_state.old_name = st.selectbox("Select curve to rename",curves, key="rename_curve_selectbox")
    # with col2:
    #     st.session_state.new_name = st.text_input("New curve name", key="rename_curve_textinput")
    # with col3:
    #     st.session_state.changename = st.button("Change", key="rename_curve_change")
    # if st.session_state.changename:
    #     st.session_state.selected_well_rename = well.rename(columns={st.session_state.old_name:st.session_state.new_name})
    #     st.dataframe(st.session_state.selected_well_rename,width=1400)

    #     #4.2_DOWNLOAD-----------------------------------------------------------------
    #     st.write("4.2 Download well curves with renamed names")
    #     st.download_button(label='Download CSV File',
    #                 data = st.session_state.selected_well.to_csv(),
    #                 file_name=f"{selected_well}.csv",
    #                 mime='text/csv')

    st.session_state.selected_well_rename = None
    selected_well_rename = st.selectbox("Select Well", list_well, key="well_selectbox")
# print("in ra gieng lua chon:", selected_well_rename)

# get the curves for the selected well
# print("las data list:",las_data_list)
# print("well name:",well_names)
# for well_1, las_file_1 in zip(well_names, las_data_list):
#     print("well", well_1)
#     print("las_file", las_file)
#     if well_1 == selected_well_rename:
#         las = las_file_1
#         break

    well_to_las = {}
    well = []
    data_rename_1 =[]
    df_all_full = pd.DataFrame()
    st.session_state.selected_well_rename = None
    for i in range(len(well_names)):
        well_to_las[well_names[i]] = las_data_list[i]
        # print("key: ",well_names[i][:len(selected_well_rename)], " value: ",las_data_list[i])
        if  selected_well_rename == well_names[i][:len(selected_well_rename)]:
            las = las_data_list[i]
            break
        # print("In ra las:", las)
    well = las.df()
    well['WELL'] = las.well.WELL.value
    # print("In ra well2:", well)
    curves = well.columns.tolist()
    # print ("print ra cuvers:", curves) # save the number of curves for the selected well in session state
    st.session_state.num_curves = len(curves)
    df_rename = pd.DataFrame()
    st.session_state.selected_well_rename = df_rename
    st.session_state.selected_well_rename = well

    # st.session_state.selected_well_rename = st.session_state.selected_well_rename.copy()
    # col1, col2, col3 = st.columns([3,3,1])
    # with col1:
    #     # update the curve dropdown options based on the selected well
    #     # st.session_state.old_name = st.text_input("Select curve to rename", key="rename_curve_selectbox")
    #     st.session_state.old_name = st.selectbox("Select curve to rename", curves, key="rename_curve_selectbox")
    # with col2:
    #     st.session_state.new_name = st.text_input("New curve name", key="rename_curve_textinput")
    # with col3:
    #         # st.session_state.changename = st.button("Change", key="rename_curve_change")
    # # if st.session_state.changename:
    #         if 'count' not in st.session_state:
    #             st.session_state.selected_well_rename2 = st.session_state.selected_well_rename.copy()
    #         if st.button('Rename'):
    #             st.session_state.selected_well_rename2 = st.session_state.selected_well_rename2.rename(columns={st.session_state.old_name:st.session_state.new_name})
    #         st.session_state.selected_well_rename2["DEPTH"] = well.index

    # st.session_state.selected_well_rename2.insert(0, 'DEPTH', st.session_state.selected_well_rename2.pop('DEPTH'))
    # st.session_state.selected_well_rename2 = st.session_state.selected_well_rename2.reset_index(drop=True)
    # st.session_state.selected_well_rename2.index.name = 'DEPTH'
    
    # st.dataframe(st.session_state.selected_well_rename2, width=1400)
    # st.session_state.selected_well_rename2.to_csv('sample.csv', index=False)
import pandas as pd

# Khởi tạo DataFrame từ dữ liệu có sẵn
# data = pd.read_csv('sample.csv')
# Khởi tạo thuộc tính selected_well_rename trong st.session_state
st.session_state.setdefault('selected_well_rename', 'Default value')
# Truy cập thuộc tính selected_well_rename

data = st.session_state.selected_well_rename 
# import streamlit as st

# # Hiển thị bảng và cho phép chỉnh sửa tên cột
# # Hiển thị bảng và cho phép chỉnh sửa tên cột
# n_cols = len(data.columns)
# cols = st.columns(n_cols)
# new_columns = []
# for i, col in enumerate(data.columns):
#     new_col = cols[i].text_input(f"Enter new name for '{col}'", col)
#     new_columns.append(new_col)

# # Đổi tên cột và hiển thị lại bảng
# data.columns = new_columns
# st.dataframe(data)

# Hiển thị bảng và cho phép chỉnh sửa tên cột
n_cols = 4
n_rows = -(-len(data.columns) // n_cols)  # Round up division
for i in range(n_rows):
    cols = st.columns(n_cols)
    for j in range(n_cols):
        idx = i * n_cols + (j-1)
        if idx < len(data.columns):
            col = data.columns[idx]
            
            # Lưu trữ tên cũ trong biến old_col
            old_col = col
            # new_col == well_names[i][:len(col)]
            new_col = cols[j-1].text_input(f"Enter new name for '{col}'", key=f"input_{cols}")

            # Kiểm tra nếu người dùng không nhập tên mới
            if not new_col:
                # Sử dụng tên cũ thay thế
                new_col = old_col
            data = data.rename(columns={col: new_col})
            data["DEPTH"] = well.index
            data.insert(0, 'DEPTH', data.pop('DEPTH'))
            data = data.reset_index(drop=True)

# Hiển thị lại bảng
# data = data.rename(columns={'': 'Unnamed'})
st.dataframe(data)
def my_function(data):
    result = None
    # Lưu trữ DataFrame khi người dùng nhấn vào nút "Lưu"
    if st.button("Lưu", key="saved_rename"):
        # Tạo tên file CSV dựa trên biến selected_well_rename
        file_name = f"{selected_well_rename}.csv"
        # Lưu trữ DataFrame vào tệp CSV với tên file tương ứng
        data.to_csv(file_name, index=False)
        result = data.copy()
        # print ("result", result)
    else:
        print("Button not clicked")
    # Trả về giá trị của biến result
    return result
        # print ("result", result)
    # Trả về giá trị của biến result
    return result 
result = my_function(data)
print("resultssss",result)
for name in selected_well_multi:
    dataframes.append(result[result.WELL==name])
        # 4.2_DOWNLOAD-----------------------------------------------------------------
st.write("4.2 Download well curves with renamed names")
st.download_button(label='Download CSV File',
            data = data.to_csv(),
            file_name=f"{selected_well_rename}.csv",
            mime='text/csv')
# list_well_1 = list_well.copy()
# if 'data' not in st.session_state:
    # st.session_state.data = result
# result = data.copy()
selected_well_multi= st.multiselect(" 4.3 Select well for download", list_well, key = 'selected_well_multi_last')  
# dowload_dataframes_df = pd.DataFrame()
st.session_state.changename_download = st.button("Create", key="selected_well_multi_rename_curve_111")
# st.session_state.selected_well_multi_11 = []
if st.session_state.changename_download:
    # dataframes = []

    dataframes_df = pd.concat(result, ignore_index=True)
    st.session_state.selected_well_multi = dataframes_df.loc[dataframes_df['WELL'].isin(selected_well_multi)].reset_index(drop=True)
    # for name in selected_well_multi:
    #     dataframes.append(result[result.WELL==name])

    # if dataframes:
    #     dowload_dataframes_df = pd.concat(dataframes, ignore_index=True).reset_index(drop=True)
        # print("in ra", dowload_dataframes_df)
    # else:
    #     print("No objects to concatenate")
    st.dataframe(dataframes_df, width=1400)

