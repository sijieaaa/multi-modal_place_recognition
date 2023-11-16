

import torch.nn as nn
import torch
import torch.nn.functional as F


from tools.utils import set_seed
from tools.options import Options
args = Options().parse()
set_seed(7)



class MetricTensor(nn.Module):
    def __init__(self, hidden_features):
        super().__init__()
        self.M = torch.zeros([hidden_features,hidden_features],dtype=torch.float32,requires_grad=True)
        self.M = nn.Parameter(self.M, requires_grad=True)

    def forward(self, h, h_transpose):
        out = h  @ self.M @ h_transpose

        return out







class GraphAttentionLayer(nn.Module):

    # graph attention layer with learnable matrix

    def __init__(self, 
                in_features, 
                hidden_features,
                n_heads=8,
                learn_m=False
                ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.n_heads = n_heads
        self.learn_m = learn_m


        self.W_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()

        if learn_m:
            self.M = MetricTensor(hidden_features)
        

        self.W = nn.Linear(in_features,in_features,True)


        if args.gattnorm == 'ln':
            self.norm = nn.LayerNorm(in_features)
        if args.gattactivation == 'gelu':
            self.activation = nn.GELU()
        if args.gattactivation == 'relu':
            self.activation = nn.ReLU(True)





    def forward(self, h):


        b, num_centroids, c = h.shape

        h_org = h


        h = self.W(h_org)

        h = h.view(b, num_centroids, self.n_heads, -1)
        h = h.permute(0,2,1,3) # [b, n_heads, num_centroids, c]

        if self.learn_m:
            attention = self.M(h, h.transpose(-2,-1)) 
        elif not self.learn_m:
            attention = h @ h.transpose(-2,-1)


        attention = F.softmax(attention, -1)
        h = attention @ h

        out = h
        out = out.permute(0,2,1,3)
        out = out.contiguous()
        out = out.view(b, num_centroids, -1)
        


        if args.gattnorm is not None:
            out = self.norm(out)

        if args.gattactivation is not None:
            out = self.activation(out)


     

        return out