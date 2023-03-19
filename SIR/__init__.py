
import streamlit as st

import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd


from .utils import get_SIR_data, vizualise_data, layer_config
from .model import DataSIR, train, test
from .data_manager import get_model, save_model




def func_SIR():
     """
    This function generates a SIR model with a Neural ODE, allowing the user to choose the parameters for the model and the architecture used for the latent ODE.
    The user can select the parameters of the SIR model, as well as the number of evolutions used for training.
    The function also visualizes the generated data and allows the user to choose the model parameters for the neural ODE.
    
    Returns:
    None
    """
    st.markdown("# SIR model")
    st.write("This app try to fit the evolution of a SIR model with a Neural ODE")

    # parameters for SIR data generation
    with st.expander("SIR data parameters") :

        st.write("Where you can choose the parameters of the SIR model, as well as the number of evolution used for training")
        col_S0, col_I0, col_R0 = st.columns(3)

        S0 = col_S0.number_input("Susceptible", 0, 1000, 100, 1, format="%d")
        I0 = col_I0.number_input("Infected", 0, 1000, 1, 1, format="%d")
        R0 = col_R0.number_input("Recovered", 0, 1000, 0, 1, format="%d")


        col_beta, col_gamma = st.columns(2)

        beta = col_beta.slider("Infection rate", 0., 1., 0.5, 0.01, format="%.2g")
        gamma = col_gamma.slider("Recovery rate", 0., 1., 0.05, 0.01, format="%.2g")

        col_nDays, col_noiseStd, col_train_size = st.columns(3)
        
        n_days = col_nDays.number_input("Num days", 1, 1000, 100, 1, format="%d")
        noise_std = col_noiseStd.number_input("Noise std", 0., 1., 0.0, 0.001, format="%.3g")
        train_size = col_train_size.number_input("train size ", 1, 1000, 1, 1, help="use 'train size' number of complete (noised) evolutions of the S, I, R populations")


    # getting data with those parameters

    n_data_per_day=10 #nb de points par jour
    t, evol_ODE, data_in = get_SIR_data(S0, I0, R0, beta, gamma, n_days, noise_std, train_size+1, n_data_per_day)
    data_train_in, data_test_in = data_in[:-1], data_in[-1]

    fig = vizualise_data(t, evol_ODE, data_test_in, name_data='test sample')

    st.pyplot(fig)



    # parameters for the neural ODE param
    with st.expander("model parameters") :
        st.markdown("Where you can choose the parameters for the model and the architecture uses for the latent ODE")
        st.markdown("""Note that if all is selected, the model is fed with $x_{t_0}$ and predict $x_{t+1}, \ldots, x_{t_f}$  
        Otherwise the model is fed with $x_{t}$ and predict $x_{t+step}$ for $t \in \{t_0, \ldots, t_{f-step}\}$
                    """)

        col_latent_ODE_selec, col_step, col_by_all = st.columns([3, 1, 1])
        latent_ODE_selec = col_latent_ODE_selec.selectbox('latent ODE model', ['default', 'minimal', 'custom'])
        by_all = col_by_all.checkbox('all')
        by_step = col_step.number_input('step', 1, 50, 1, 1, disabled=by_all)


        col_num_epoch, col_loss_fn_name, col_optimizer_name, col_lr, col_use_t = st.columns(5)
        
        num_epoch       = col_num_epoch.number_input("Num epoch", 0, 1000, 50, 1, format="%d")
        loss_fn_name    = col_loss_fn_name.selectbox("loss function", ['MSELoss'])
        optimizer_name  = col_optimizer_name.selectbox("optimizer", ['Adam', 'SGD'])
        lr              = col_lr.number_input("learning rate", 0., 1., 0.001, 0.001, format="%.3f")  
        use_t           = col_use_t.checkbox("use t", False, help="if checked the latent ODE take t as an input")        

        st.markdown("Time used for the integration : (The model is fed with $x_{t_0}$ and predicts $x_{t_0+1},\ldots,x_{t_f}$.)")
        
        col_t0_train, col_tf_train, col_t0_test, col_tf_test, _ = st.columns(5)
        t0_train        = col_t0_train.number_input('$t_0$ train', 0, n_days, 0, 1, key='t0_train')
        tf_train        = col_tf_train.number_input('$t_f$ train', 0, n_days, n_days, 1, key='tf_train')
        t0_test         = col_t0_test.number_input('$t_0$ test', 0, n_days, 0, 1, key='t0_test')
        tf_test         = col_tf_test.number_input('$t_f$ test', 0, n_days, n_days, 1, key='tf_test')

        in_size = 3 if not use_t else 4

        if latent_ODE_selec == 'default' : 
            latent_ODE_net = nn.Sequential(nn.Linear(in_size, 10),
                                            nn.ReLU(),
                                            nn.Linear(10, 3))        
        elif latent_ODE_selec == 'minimal' : 
            latent_ODE_net = nn.Sequential(nn.Linear(3, 3))

        elif latent_ODE_selec == 'custom' :
            num_layers = st.slider("Num Layers", 0, 10, 2, 1, help="if the last layer is not of size 3, it will add an Linear(last_size, 3) to the model")

            layers = []
            for i in range(num_layers) :
                layer, in_size = layer_config(i+1, in_size)
                layers.append(layer)

            if in_size != 3 or len(layers) == 0 :
                layers.append(nn.Linear(in_size, 3))

            latent_ODE_net = nn.Sequential(*layers)


    st.write("latent ODE model :")
    st.code(latent_ODE_net)
    

    params_dict = {'S0':S0, 
                   'I0':I0, 
                   'R0':R0, 
                   'beta':beta, 
                   'gamma':gamma,
                   'n_days':n_days, 
                   'noise_std':noise_std, 
                   'train_size':train_size,
                   'by_step':by_step,
                   'by_all':by_all,
                   'num_epoch':num_epoch, 
                   'loss_fn_name':loss_fn_name, 
                   'optimizer_name':optimizer_name, 
                   'lr':lr,
                   'use_t':use_t,
                   't0_train':t0_train, 
                   'tf_train':tf_train, 
                   'latent_ODE_net':latent_ODE_net}


    if not 'model_SIR' in st.session_state.keys():
        succes, model = get_model(params_dict)
        if succes :
            st.success('default model loaded')
        else :
            st.error('default model not found, forcing default training (if you change the parameters now it will become the default model)')  
            step = by_step if not by_all else 'all' 
            loader = DataLoader(DataSIR(data_train_in, n_data_per_day, t0_train, tf_train, step))
            # st.write(len(loader))
            # st.write(len(loader.dataset))
            with st.empty() :
                model = train(loader, num_epoch, loss_fn_name, optimizer_name, lr, use_t,
                            latent_ODE_net)
                st.success('default training completed')

            save_model(model, params_dict, 'default')
        st.session_state['model_SIR'] = model
        
    else : 
        model  = st.session_state['model_SIR']

    if st.button('Update model'):
        # update model with interactive inputs
        succes, model = get_model(params_dict)
        with st.empty() :
            if succes :
                st.success('model loaded')
            else :
                step = by_step if not by_all else 'all'  
                st.code(step)             
                loader = DataLoader(DataSIR(data_train_in, n_data_per_day, t0_train, tf_train, step))
                model = train(loader, num_epoch, loss_fn_name, optimizer_name, lr, use_t,
                            latent_ODE_net)
                save_model(model, params_dict)

        st.session_state['model_SIR'] = model



    if st.button('Update prediction'):
        step = by_step if not by_all else 'all'              

        test_loader = DataLoader(DataSIR(data_test_in, 10, t0_test, tf_test, step))
        loss, preds = test(test_loader, model, loss_fn_name)

        if by_all :
            preds = preds[0]
            idx_t0 = t0_test * n_data_per_day + 1
            idx_tf = (tf_test)*n_data_per_day 
        else :
            idx_t0 = (t0_test+step)*n_data_per_day
            idx_tf = (tf_test)*n_data_per_day 
        

        fig = vizualise_data(t, evol_ODE, preds, idx_t0, idx_tf, name_data='prediction')

        st.pyplot(fig)

    from .data_manager import available_models, _update_available

    _update_available()

    st.write("Where are all the models saved on disk and there parameters :")
    st.dataframe(pd.DataFrame(available_models).T)

     
