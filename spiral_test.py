import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import streamlit as st





#This code is from RTQ Chen [4]. The docstrings are from us but we wanted to include his example in our app.




def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=False):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


class LatentODEfunc(nn.Module):
    """ 
LatentODEfunc: 
    This module defines the neural network architecture for the latent ODE function that learns the dynamics of the system. The forward function takes as input the time and the current state of the system, and outputs the derivative of the state with respect to time. The architecture consists of three fully connected layers with ELU activation function.

    Parameters:
        latent_dim (int): The dimension of the latent space.
        nhidden (int): The number of hidden units in the neural network.

    Attributes:
        elu (nn.Module): The ELU activation function.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        nfe (int): The number of function evaluations (i.e., forward passes) that have been performed.

    Methods:
        forward(t, x): Computes the derivative of the state with respect to time, given the time and the current state of the system.
    """

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    """RecognitionRNN:
    This module defines the neural network architecture for the recognition model that infers the latent state from the observed data. The forward function takes as input the observed data and the hidden state, and outputs the mean and variance of the posterior distribution over the latent state. The architecture consists of two fully connected layers with tanh activation function.

    Parameters:
        latent_dim (int): The dimension of the latent space.
        obs_dim (int): The dimension of the observed data.
        nhidden (int): The number of hidden units in the neural network.
        nbatch (int): The batch size.

    Attributes:
        i2h (nn.Linear): The input-to-hidden layer.
        h2o (nn.Linear): The hidden-to-output layer.
        nhidden (int): The number of hidden units in the neural network.
        nbatch (int): The batch size.

    Methods:
        forward(x, h): Computes the mean and variance of the posterior distribution over the latent state, given the observed data and the hidden state.
        initHidden(): Initializes the hidden state with zeros."""

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):
    """
    A neural network decoder module.

    Args:
        latent_dim (int): The dimension of the latent space.
        obs_dim (int): The dimension of the observation space.
        nhidden (int): The number of hidden units in the fully-connected layers.

    Attributes:
        relu (nn.ReLU): The ReLU activation function.
        fc1 (nn.Linear): The first fully-connected layer, mapping from the latent space to the hidden layer.
        fc2 (nn.Linear): The second fully-connected layer, mapping from the hidden layer to the observation space.

    Methods:
        forward(z: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the decoder network.

            Args:
                z (torch.Tensor): The input tensor of shape (batch_size, latent_dim).

            Returns:
                The output tensor of shape (batch_size, obs_dim).
    """

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    """
    Computes the log pdf of a normal distribution with mean `mean` and log variance `logvar` at the point `x`.

    Args:
        x (torch.Tensor): The point(s) at which to compute the log pdf. Should have shape (batch_size, dim).
        mean (torch.Tensor): The mean of the normal distribution. Should have shape (batch_size, dim).
        logvar (torch.Tensor): The log variance of the normal distribution. Should have shape (batch_size, dim).

    Returns:
        torch.Tensor: The log pdf of the normal distribution evaluated at `x`. Should have shape (batch_size,).
    """
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """
    Computes the KL divergence between two normal distributions with means `mu1` and `mu2` and log variances `lv1` and `lv2`.

    Args:
        mu1 (torch.Tensor): The mean of the first normal distribution. Should have shape (batch_size, dim).
        lv1 (torch.Tensor): The log variance of the first normal distribution. Should have shape (batch_size, dim).
        mu2 (torch.Tensor): The mean of the second normal distribution. Should have shape (batch_size, dim).
        lv2 (torch.Tensor): The log variance of the second normal distribution. Should have shape (batch_size, dim).

    Returns:
        torch.Tensor: The KL divergence between the two normal distributions. Should have shape (batch_size,).
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl





def main_spiral_chen(): 
    """Trains a Latent Ordinary Differential Equation model on a toy spiral dataset and visualizes the learned trajectory.

    The model consists of a LatentODEfunc, RecognitionRNN, and Decoder. The training procedure involves
    infering the initial distribution of the latent space by moving backwards in time through the RecognitionRNN,
    and then moving forwards in time to learn a trajectory through the LatentODEfunc and Decoder. The loss is
    calculated as the sum of the negative log-likelihood of the observations and the KL divergence between the approximate
    posterior and the prior. 

    Returns:
    - None, but displays the learned trajectory and a plot of the loss during training using Streamlit.
    """
    st.write("The training is quite long, please consider a coffee break ([4]). We also added some docstrings on the original code.")

    col1, col2, col3 = st.columns(3)
    adjoint = col1.checkbox('use adjoint', True)
    niters = col2.number_input('n iters', 30, 5000, 2000, 1)
    lr = col3.number_input('learning rate', 0., 1., 0.01, 0.0001)



    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint



    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()

    with st.empty() :
        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            st.write('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

        st.write('Training complete after {} iters.'.format(itr))


    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        h = rec.initHidden().to(device)
        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        orig_ts = torch.from_numpy(orig_ts).float().to(device)

        # take first trajectory for visualization
        z0 = z0[0]

        ts_pos = np.linspace(0., 2. * np.pi, num=2000)
        ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        ts_neg = torch.from_numpy(ts_neg).float().to(device)

        zs_pos = odeint(func, z0, ts_pos)
        zs_neg = odeint(func, z0, ts_neg)

        xs_pos = dec(zs_pos)
        xs_neg = torch.flip(dec(zs_neg), dims=[0])

        xs_pos = xs_pos.cpu().numpy()
        xs_neg = xs_neg.cpu().numpy()
        orig_traj = orig_trajs[0].cpu().numpy()
        samp_traj = samp_trajs[0].cpu().numpy()

        fig, ax = plt.subplots(1,1)
        ax.plot(orig_traj[:, 0], orig_traj[:, 1],
                 'g', label='true trajectory')
        ax.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                 label='learned trajectory (t>0)')
        ax.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
                 label='learned trajectory (t<0)')
        ax.scatter(samp_traj[:, 0], samp_traj[
                    :, 1], label='sampled data', s=3)
        ax.legend()

        st.pyplot(fig)
