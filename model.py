import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str
    

class ConvNet_avgPool_share(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, avgPoolLen, n_channs, stratified, activ, phase):
        super(ConvNet_avgPool_share, self).__init__()
        # self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen))
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.stratified = stratified
        self.activ = activ
        assert phase in ['train', 'infer']
        self.phase = phase

    def forward(self, input):
        if 'initial' in self.stratified:
            if self.phase == 'train':
                input = stratified_layerNorm(input, int(input.shape[0]/2))
            elif self.phase == 'infer':
                input = stratified_layerNorm(input, int(input.shape[0]))
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        
        if self.activ == 'square':
            out = self.avgpool(out ** 2)
        elif self.activ == 'logvar':
            out = torch.log(self.avgpool(out ** 2) + 1e-5)
        elif self.activ == 'relu':
            out = self.avgpool(F.relu(out))
        if 'middle' in self.stratified:
            if self.phase == 'train':
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            elif self.phase == 'infer':
                out = stratified_layerNorm(out, int(out.shape[0]))
        out = out.reshape(out.shape[0], -1)
        return out
    
class ConvNet_avgPool_share_nopool(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, n_channs, stratified, phase):
        super(ConvNet_avgPool_share_nopool, self).__init__()
        # self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen))
        self.stratified = stratified
        assert phase in ['train', 'infer']
        self.phase = phase

    def forward(self, input):
        if 'initial' in self.stratified:
            if self.phase == 'train':
                input = stratified_layerNorm(input, int(input.shape[0]/2))
            elif self.phase == 'infer':
                input = stratified_layerNorm(input, int(input.shape[0]))
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        
        if 'middle' in self.stratified:
            if self.phase == 'train':
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            elif self.phase == 'infer':
                out = stratified_layerNorm(out, int(out.shape[0]))
        out = out.reshape(out.shape[0], -1)
        return out