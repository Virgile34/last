import streamlit as st
import base64
import requests
from streamlit.components.v1 import html
import webbrowser

def func_intro() :
    # Set page title
    # st.set_page_config(page_title="Neural ODE Intro", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
    # Set page background color
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #A9A9A9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Define CSS style
    css = """
    h1 {
        color: #1F1F1F;
        font-family: 'Roboto', sans-serif;
        font-size: 3rem;
        text-align: center;
        margin-top: 2rem;
    }

    h2 {
        color: #333333;
        font-family: 'Roboto', sans-serif;
        font-size: 2rem;
        margin-top: 1.5rem;
    }

    p {
        color: #333333;
        font-family: 'Roboto', sans-serif;
        font-size: 1.2rem;
        line-height: 1.5;
        margin-top: 1rem;
    }

    .container {
        max-width: 800px;
        margin: auto;
        padding: 2rem;
        background-color: #f2f2f2;
        border-radius: 5px;
    }
    """

    # Add CSS to page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    # Page title and introduction
    st.title("Introduction to Neural ODEs")
    st.write("Neural ODE is a powerful concept in deep learning introduced in 2018 that has gained a lot of attention in recent years. It allows to model continuous-time dynamics using neural networks, enabling us to solve a wide range of problems in fields like physics, biology, and finance. On this app, please use the sidebar, to play with different models.")

    # Add a picture
    image_url = "https://drive.google.com/uc?id=11sR57qafL6ks0shW1OE0wATExG8m4lWt"
    st.image(image_url, caption="")
    st.write("Here is (in french) our short introduction to Neural ODEs:")

    # Embed PDF
    pdf_url = "https://drive.google.com/file/d/1mY01qXKj0HOFW2WEvXoWGO4oN5-VSQBa/preview"
    height = 600
    html_string = f'<iframe src="{pdf_url}" width="100%" height="{height}px"></iframe>'
    st.markdown(html_string, unsafe_allow_html=True)
    st.write("Here are (in french) the slides we used for our presentation:")
    # Embed PDF slides
    pdf_url = "https://drive.google.com/file/d/1DGUX41ey6n0jqZ_-3TLkihcRkLNglawZ/preview"
    height = 600
    html_string = f'<iframe src="{pdf_url}" width="100%" height="{height}px"></iframe>'
    st.markdown(html_string, unsafe_allow_html=True)

    # Add links to relevant articles
    st.write("Here are some relevant references on which we based our work for this app:")
    st.write("[1] Neural-ODE for pharmacokinetics modeling and its advantage to alternative machine learning models in predicting new dosing regimens (LU, 2021)- https://www.sciencedirect.com/science/article/pii/S2589004221007720")
    st.write("[2] Neural Ordinary Differential Equations (KIDGER, 2019) - https://arxiv.org/abs/1806.07366")
    st.write("[3] Neural Differential Equations for Supervised Learning (CHEN, 2018) - https://arxiv.org/abs/2202.02435")
    st.write("[4] https://github.com/rtqichen")