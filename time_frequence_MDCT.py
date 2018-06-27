#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:50:59 2018

@author: shixuan
"""

from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.signal


class ifft(nn.Module):
    def __init__(self, nfft=1024):
        super(ifft, self).__init__()
        assert nfft % 2 == 0
        self.nfft = int(nfft)
        self.n_freq = n_freq = int(nfft / 2)
        real_kernels, imag_kernels, self.ac_cof = _get_ifft_kernels(nfft)
        self.real_conv = nn.Conv2d(1, nfft, (n_freq, 1), stride=1, padding=0, bias=False)
        self.imag_conv = nn.Conv2d(1, nfft, (n_freq, 1), stride=1, padding=0, bias=False)

        self.real_conv.weight.data.copy_(real_kernels)
        self.imag_conv.weight.data.copy_(imag_kernels)
        self.real_model = nn.Sequential(self.real_conv)
        self.imag_model = nn.Sequential(self.imag_conv)
    def forward(self, magn, phase, ac=None):
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        output = self.real_model(magn) - self.imag_model(phase)
        if ac is not None:
            output = output + ac * self.ac_cof
        return output / self.nfft






def _get_ifft_kernels(nfft):
    nfft = int(nfft)
    assert nfft % 2 == 0
    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / 1024.)

    kernels = np.fromfunction(kernel_fn, (int(nfft), int(nfft/2+1)), dtype=np.float64)

    kernels = np.zeros((1024, 513)) * 1j

    '''
    for i in range(1024):
        for j in range(513):
            kernels[i, j] = kernel_fn(i, j)
    '''

    ac_cof = float(np.real(kernels[0, 0]))

    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)

    real_kernels = torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis])
    imag_kernels = torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis])
    return real_kernels, imag_kernels, ac_cof




class imdct(nn.Module):
    def __init__(self, nfft=512, hop_length=256):
        super(imdct, self).__init__()
        assert nfft % 2 == 0
        assert hop_length <= nfft
        self.hop_length = hop_length

        self.nfft = int(nfft)
        self.n_freq = n_freq = int(nfft / 2)
        self.kernels = _get_imdct_kernels(nfft)
        trans_kernels = np.zeros((nfft, nfft), np.float64)
        np.fill_diagonal(trans_kernels, np.ones((nfft, ), dtype=np.float64))
        # self.win_cof = 1 / scipy.signal.get_window("hanning", nfft)
        # self.win_cof[0] = 0
        # self.win_cof = torch.from_numpy(self.win_cof).float()
        # self.win_cof = 11nn.Parameter(self.win_cof, requires_grad=False)
        self.trans_kernels = nn.Parameter(torch.from_numpy(trans_kernels[:, np.newaxis, np.newaxis, :]).float())

    def forward(self, spec):
        '''
        batch None frequency frame
        '''
        assert spec.size()[2] == self.n_freq
        nfft = self.nfft
        hop = self.hop_length

        # complex conjugate
        output = F.conv2d(spec, self.kernels)

        output = output * 4 / float(self.nfft)

        output = F.conv_transpose2d(output, self.trans_kernels, stride=self.hop_length)
        output = output.squeeze(1)
        output = output.squeeze(1)
        # output[:, :hop] = output[:, :hop].mul(self.win_cof[:hop])
        # output[:, -(nfft - hop):] = output[:, -(nfft - hop):].mul(self.win_cof[-(nfft - hop):])
        return output

def _get_imdct_kernels(nfft):
    nfft = int(nfft)
    assert nfft % 2 == 0
    def kernel_fn(time, freq):
        win_cof = np.sin(np.pi*(2*time+1)/(2*float(nfft)))
        return np.cos(np.pi/float(nfft)*(2*time+1+float(nfft)/2)*(freq+1/2))*win_cof
    kernels = np.fromfunction(kernel_fn, (int(nfft), int(nfft//2)), dtype=np.float64)
    
    kernels = nn.Parameter(torch.from_numpy(kernels[:, np.newaxis, :, np.newaxis]).float())

    return kernels



class mdct(nn.Module):
    def __init__(self, nfft=512, hop_length=256):
        super(mdct, self).__init__()
        assert nfft % 2 == 0

        self.hop_length = hop_length
        self.n_freq = n_freq = nfft//2 + 1

        self.kernels = _get_mdct_kernels(nfft)

    def forward(self, sample):
        sample = sample.unsqueeze(1)
        sample = sample.unsqueeze(1)

        spec = F.conv2d(sample, self.kernels, stride=self.hop_length)

        spec = spec.permute(0, 2, 1, 3)

        return spec


def _get_mdct_kernels(nfft):
    nfft = int(nfft)
    assert nfft % 2 == 0

    def kernel_fn(freq, time):
        win_cof = np.sin(np.pi*(2*time+1)/(2*float(nfft)))
        return np.cos(np.pi/float(nfft)*(2*time+1+float(nfft)/2)*(freq+1/2))*win_cof

    kernels = np.fromfunction(kernel_fn, (int(nfft//2), int(nfft)), dtype=np.float64)
 
    kernels = nn.Parameter(torch.from_numpy(kernels[:, np.newaxis, np.newaxis, :]).float())

    return kernels



if __name__ == "__main__":
    signal = np.random.rand(1024 * 10)
    signal = signal - np.mean(signal)
    signal = signal[np.newaxis, :]
    model = stft(window="retangle")
    real, imag, ac = model.forward(Variable(torch.from_numpy(signal).float()))
    real = real.data.numpy()
    imag = imag.data.numpy()
    ac = ac.data.numpy()
    print(ac)


