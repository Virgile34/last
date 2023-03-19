import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchdiffeq import odeint, odeint_adjoint

class _latentODE(nn.Module) :
    def __init__(self, net, use_t=False) -> None:
        super(_latentODE, self).__init__()
        self.net = net
        self.use_t = use_t

    def forward(self, t, x) :
        if self.use_t :
            return self.net(torch.cat((x, t)))
        else :
            return self.net(x)


class SIR_NeuralODE(nn.Module):
    """
    The NeuralODE class for SIR model:
        SIR_NeuralODE(X, integration_time) return odeint(func, X, int)
        !Warning : does not accept batched input!

    :inherits: nn.Module

    Attributes:
    -----------
    net: nn.Sequential
        The model used to estimate dy/dt
    use_t: bool, default=False
        if True the model used to estimate dy/dt take as an input (x_t, t) and not only x_t 
        (hence the input dim is dim_data+1 if True)

    adjoint: bool, default=False
        if True use odeint_adjoint, otherwise use odeint (from torchdiffeq)

    Params:
    -------
    x : nn.Tensor of shape (dim_data, )
        the value at t0
    integration_time : nn.Tensor of shape (pred_len+1, )
        the integration period, sorted, where the first value is t0

    Returns:
    --------
    nn.Tensor of shape (pred_len, dim_data)
    the prediction of x_t over integration_time[1:]

    """
    def __init__(self, net, use_t=False, adjoint=False):
        super(SIR_NeuralODE, self).__init__()

        self.odefunc = _latentODE(net, use_t)
        self.adjoint = adjoint

    def forward(self, x, integration_time):
        """
        x : Tensor of shape (dim_data, )
        integration_time : Tensor of shape (len_pred+1, ) 

        """
        # st.code(x)
        # st.code(integration_time)

        if self.adjoint :
            out = odeint(self.odefunc, x, integration_time)
        else :
            out = odeint_adjoint(self.odefunc, x, integration_time)
        return out[1:]


class DataSIR(Dataset):
    def __init__(self, data:np.ndarray, n_data_per_day, t0=0, tf=-1, step='all'):
        super(DataSIR, self).__init__()

        self.data = torch.DoubleTensor(data)

        if len(self.data.shape) == 2 :
            self.data = self.data.unsqueeze(0)
        
        self.step = step if step=='all' else step*n_data_per_day 
  
        self.n_data_day = n_data_per_day
        self.num_evol, self.len_evol, _ = self.data.shape # dim_at_t for the _


        self.t0 = t0*n_data_per_day
        if tf == -1 : self.tf = self.len_evol
        else: self.tf = tf*n_data_per_day + 1
        # st.write(self.len_evol)
        # st.write(self.tf)
        # st.write(self.t0)
        # st.write(self.tf)
        # st.write(self.len_evol)

    def set_integration_time(self, start:int, end:int) :
        self.t0 = start*self.n_data_day
        self.tf = end*self.n_data_day + 1

    def __len__(self):
        if self.step == 'all' :
            return self.num_evol
        else : 
            return self.num_evol * (self.tf-self.t0 - self.step)
        
    def __getitem__(self, idx):
        if self.step == 'all' :
            x0  = self.data[idx, self.t0          , :]
            x_t = self.data[idx, self.t0+1:self.tf, :]

            t = torch.DoubleTensor(range(self.t0, self.tf)) / self.n_data_day / 10

            return (x0, t), x_t

        else :
            idx_evol = idx % self.num_evol
            idx_t0 = idx - idx // (self.tf-self.t0 - self.step)

            # st.write(idx)
            # st.write(self.num_evol)
            # st.write(self.len_evol - self.step)
            # st.write(idx // (self.len_evol - self.step))
            # st.write(str(idx_t0) + "," + str(idx_evol))

            x0  = self.data[idx_evol, self.t0+idx_t0, :]
            x_t = self.data[idx_evol, self.t0+idx_t0+self.step, :]

            t = torch.DoubleTensor([idx_t0, idx_t0+self.step]) / self.n_data_day
            
            return (x0, t), x_t

def train(loader, num_epoch, loss_fn_name, optimizer_name, lr, use_t, latent_ODE_net) :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = SIR_NeuralODE(latent_ODE_net, use_t, adjoint=False)
    loss_fn = getattr(nn, loss_fn_name)()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # model.train()
    for i in range(num_epoch) : 
        with st.empty().container() :
            info = 'training epoch {} / {}'.format(i+1, num_epoch)
            if i != 0 : info += ', last loss : {}'.format(loss)

            st.write(info)
            progress_bar = st.progress(0)
            loss = _train_one_epoch(loader, model, loss_fn, optimizer, device, progress_bar)

    return model

def _train_one_epoch(dataloader, model, loss_fn, optimizer, device, progress_bar=None):
    """The training step for one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The training DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    optimizer: torch.optim.optimizer.Optimizer
        The optimizer.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    progress_bar: st.progress
        if provided the training progress is shown with the progress bar
        otherwise prints to the console
    text : str
        the text of the progress bar

    Returns
    -------
    train_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_loss = 0

    model.train()

    for batch, ((x0, t), x_t) in enumerate(dataloader) :
        # st.write(batch)
        # st.write(x0.shape)
        # st.write(t.shape)
        # st.write(x_t.shape)

        x0, t, x_t = x0.to(device), t.to(device), x_t.to(device)
        
        # Compute prediction
        pred = torch.zeros(x_t.shape, dtype=x_t.dtype)
        for i in range(len(pred)) :
            # st.write("x0 : {}, t : {}".format(x0, t))
            pred[i] = model(x0[i], t[i])


        loss = loss_fn(pred, x_t)
        batch_loss = loss.item()
        train_loss += batch_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if progress_bar != None :
            progress_bar.progress((batch+1) / num_batches)
        else :
            print("batch {} / {}".format(batch+1, batch))
    train_loss /= num_batches
    return train_loss




# def test_step(dataloader, model, loss_fn_name, device=None):
#     if device is None :
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     loss_fn = getattr(nn, loss_fn_name)()

#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()

#     test_loss = 0

#     all_preds = np.zeros((len(dataloader.dataset), 3))
#     with torch.no_grad():
#         idx = 0
#         for (x0, t), x_t in dataloader:
#             x0, t, x_t = x0.to(device), t.to(device), x_t.to(device)

#             preds = torch.zeros(x_t.shape, dtype=x_t.dtype)
#             for i in range(len(x0)) :
#                 preds[i] = model(x0[i], t[i])
#                 all_preds[idx] = preds[i].cpu().detach().numpy()
#                 idx += 1

#             test_loss += loss_fn(preds, x_t).item()

#     test_loss /= num_batches
#     # st.code(all_preds)
#     return test_loss, all_preds


# @st.cache
def test(dataloader, model, loss_fn_name, device=None)  :
    if device is None :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_fn = getattr(nn, loss_fn_name)()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0

    all_preds = []
    with torch.no_grad():
        for (x0, t), x_t in dataloader:
            x0, t, x_t = x0.to(device), t.to(device), x_t.to(device)

            preds = torch.zeros(x_t.shape, dtype=x_t.dtype)
            for i in range(len(x0)) :
                preds[i] = model(x0[i], t[i])
                all_preds.append(preds[i].cpu().detach().numpy())

            test_loss += loss_fn(preds, x_t).item()

    test_loss /= num_batches
    # st.code(all_preds)
    return test_loss, np.array(all_preds)
