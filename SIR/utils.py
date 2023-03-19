
import torch.nn as nn
import streamlit as st

import numpy as np
from scipy.integrate import odeint as odeint_sc

import matplotlib.pyplot as plt


@st.cache_data
def get_SIR_data(S0, I0, R0, beta, gamma, n_days, noise_std, size, n_data_per_day=10):
    """
    Returns
    -------
    t : np.ndarray (10*n_days, )
    evol_ODE: np.ndarray (10*n_days, 3)
        the evolution of the SIR model from ODE integration
    data: np.ndarray (size, 10*n-days, 3)
        `size` noised evolution of the SIR model

    """
    N = S0 + I0 + R0

    sir_model = lambda y, t: (-beta * y[0] * y[1] / N, 
                            beta * y[0] * y[1] / N - gamma * y[1], 
                            gamma * y[1])

    t = np.linspace(0, n_days, n_data_per_day*n_days +1)
    
    evol_ODE = odeint_sc(sir_model, [S0, I0, R0], t)
    evol_ODE /= N
    # evol_ODE = evol_ODE[np.round(t) == t]

    data = np.repeat(evol_ODE[np.newaxis, :], size, axis=0)
    data += np.random.normal(0, noise_std, data.shape)

    return t, evol_ODE, data



def vizualise_data(t, evol_SIR, data, idx_t0=0, idx_tf=-1, name_data='not specified') :
    if idx_tf == -1 : idx_tf = len(t)

    fig, ax = plt.subplots(1, 1)

    size_marker=2

    ax.set_title("ground truth and {}".format(name_data))

    ax.plot(t, evol_SIR[:, 0], label="Susceptible")
    ax.plot(t, evol_SIR[:, 1], label="Infected")
    ax.plot(t, evol_SIR[:, 2], label="Recovered")

    # st.write("{} , {}".format(idx_tf+1-idx_t0, data.shape))

    ax.scatter(None, None, label=name_data, color='black', s=size_marker)
    ax.scatter(t[idx_t0:idx_tf+1], data[:, 0], s=size_marker)
    ax.scatter(t[idx_t0:idx_tf+1], data[:, 1], s=size_marker)
    ax.scatter(t[idx_t0:idx_tf+1], data[:, 2], s=size_marker)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("population (%)")

    return fig

def layer_config(num, in_size) :
    st.markdown("---")
    st.markdown("- Layer " + str(num))
    col1, col2 = st.columns(2)

    type_layer = col1.selectbox("type", ["activation function", "Linear"], key="type_layer"+ str(num), index=num%2)

    if type_layer == "Linear" :
        col2, col3 = col2.columns(2)

        out_size = col2.number_input("out_size", 1, 100, 10, 1, key="layerSize" + str(num))

        col3.text("biased")
        bias = col3.checkbox("biased", True, key="LayerBias" + str(num), label_visibility='collapsed')

        layer = nn.Linear(in_size, out_size, bias)
        
        return layer, out_size

    elif type_layer == "activation function" :
        type_activation = col2.text_input("type of activation (named as in torch.nn)", "ReLU", key="layerActivation" + str(num))
        layer = getattr(nn, type_activation)()

        return layer, in_size
    

