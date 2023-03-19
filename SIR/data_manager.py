import pickle as pkl

from torch import load as torch_load
from torch import save as torch_save
import os

from .model import SIR_NeuralODE


data_path = os.path.join('SIR', 'data')
available_models_path = os.path.join(data_path, '_available_models.pkl')

if not os.path.isdir(data_path) :
    os.mkdir(data_path)

params_stored = ('S0', 
                 'I0', 
                 'R0', 
                 'beta', 
                 'gamma', 
                 'n_days', 
                 'noise_std', 
                 'train_size',
                 'by_step',
                 'by_all',
                 'num_epoch', 
                 'loss_fn_name', 
                 'optimizer_name', 
                 'lr',
                 'use_t',
                 't0_train', 
                 'tf_train', 
                 'latent_ODE_net')


available_models = None

if not os.path.isfile(available_models_path) :
    available_models = {}
else : 
    with open(available_models_path, 'rb') as f:
        available_models = pkl.load(f)

def _update_available() :
    to_remove = []
    for filename in available_models.keys() :
        path = os.path.join(data_path, filename)
        if not os.path.isfile(path) :
            to_remove.append(filename)
            
    for filename in to_remove :        
        available_models.pop(filename)
            



def get_model(param_dict) :
    _update_available()

    assert set(param_dict.keys()) == set(params_stored), \
           "the dict pass to get_model does not contains the same keys as params_stored"

    param_dict_str = param_dict.copy()
    for key, val in param_dict_str.items() : 
        param_dict_str[key] = str(val)
        

    for filename, param in available_models.items() :
        if param == param_dict_str :
            path_model = os.path.join(data_path, filename)

            model = SIR_NeuralODE(param_dict['latent_ODE_net'], param_dict['use_t'], adjoint=False)
            model.load_state_dict(torch_load(path_model))
            return True, model
    
    return False, None
    
# def get_model_by_name(name:str):
#     if not name.endswith('.pth'):
#         name = name + '.pth'

#     path_model = os.path.join(data_path, name)

#     if name in available_models.keys() :
#         if os.path.isfile(path_model) : 
#             model = torch_load(path_model)
#             return True, model
           
#         else : 
#             available_models.pop(name)
#             with open(available_models_path, 'wb') as f:
#                 pkl.dump(available_models, f)

#     return False, None

def save_model(model, param_dict, name_model=None) : 
    _update_available()
    
    if name_model is None :
        name_model = 'model_' + str(len(available_models) + 1) + '.pth'
    else :
        name_model = name_model + '.pth'


    param_dict_str = param_dict.copy()
    for key, val in param_dict_str.items() : 
        param_dict_str[key] = str(val)

    available_models[name_model] = param_dict_str

    with open(available_models_path, 'wb') as f:
        pkl.dump(available_models, f)

    path_model = os.path.join(data_path, name_model)
        
    model.to('cpu')
    torch_save(model.state_dict(), path_model)


