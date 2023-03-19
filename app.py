"""
The main module of the app.
"""
import streamlit as st
from SIR import func_SIR



from intro_py import func_intro
from boucing import app_bouncing_ball
from spiral3 import main_spirale

from spiral_test import main_spiral_chen

if __name__ == "__main__" :

    st.set_page_config(page_title="Neural ODE Intro", page_icon=":smiley:", initial_sidebar_state="expanded")
    
    page_names_to_funcs = {
        "-": func_intro,
        "SIR": func_SIR,
        "Shape": main_spirale,
        "Spiral R.T.Q.Chen":main_spiral_chen,
        "Bouncing Ball": app_bouncing_ball
    }

    demo_name = st.sidebar.selectbox("Choose a mode", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()

