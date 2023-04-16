import streamlit as st
from PIL import Image
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