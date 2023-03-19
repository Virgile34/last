"""
The main module of the app.
"""
# import argparse

# parser = argparse.ArgumentParser(description='neural ODE application using streamlit')

# parser.add_argument('-f', '--force_train', action='store_true', help='if True will force the training of the default SIR model')

# args = parser.parse_args()

import streamlit as st
from SIR import func_SIR


# def func_intro() :
#     st.write("This is an intro to our project")


from intro_py import func_intro
from boucing import app_bouncing_ball
from spiral3 import main_spirale

if __name__ == "__main__" :

    st.set_page_config(page_title="Neural ODE Intro", page_icon=":smiley:", initial_sidebar_state="expanded")
    
    page_names_to_funcs = {
        "-": func_intro,
        "spiral": main_spirale,
        "SIR": func_SIR,
        "bouncing ball": app_bouncing_ball
    }

    demo_name = st.sidebar.selectbox("Choose a mode", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()

