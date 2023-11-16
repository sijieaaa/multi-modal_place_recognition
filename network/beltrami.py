



import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
# from vig_pytorch.gcn_lib.torch_nn import batched_index_select

import geotorch

from tools.options import Options
from tools.utils import set_seed
set_seed(7)
args = Options().parse()



class ODEFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, x):
        output = self.func(x)
        return output




# ---- beltrami_grand

class GNN(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.fc = nn.Linear(dim, dim*2)
        self.k = k

    def forward(self, x):
        assert len(x.shape) == 4
        b,c,h,w = x.shape
        
        x = x.flatten(2)
        x = x.permute(0,2,1) # [b,hw,c]
        feat_pos = self.fc(x) # [b,hw,c*2]
        feat = feat_pos[:,:,:c] # [b,hw,c]
        pos = feat_pos[:,:,c:] # [b,hw,c]
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1,-2) # [b,hw,hw]
        topksim, topkid = torch.topk(sim, k=self.k, dim=-1) # [b,hw,k]
        # fetch topk feat
        topkid = topkid.flatten(1) # [b,hw*k] 
        topkfeat = feat[torch.arange(b).unsqueeze(-1), topkid] # [b,hw*k,c]
        topkfeat = topkfeat.view(b,h*w,self.k,c) # [b,hw,k,c]
        attn = topksim.softmax(dim=-1) # [b,hw,k]
        x = (attn.unsqueeze(-1) * topkfeat).sum(dim=-2) # [b,hw,c]
        x = x.view(b,h,w,c)
        x = x.permute(0,3,1,2) # [b,c,h,w]
        
        return x
    
class Beltrami(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        func = GNN(dim=dim, k=k)
        self.odefunc = ODEFunc(func)

    def forward(self, x):
        t = torch.tensor([0,1], dtype=torch.float32, device=x.device, requires_grad=False)
        output = odeint_adjoint(self.odefunc, x, t, 
                        method=args.odeint_method,
                        rtol=args.tol, atol=args.tol)
        output = output[-1]

        return output





# ---- beltramiv2

class GNNV2(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        # self.fc = nn.Linear(dim, dim*2)
        self.k = k

    def forward(self, feat_pos):
        assert len(feat_pos.shape) == 4
        b,c,h,w = feat_pos.shape
        

        feat_pos = feat_pos.flatten(2) # [b,2c,hw]
        feat_pos = feat_pos.permute(0,2,1) # [b,hw,2c]
        feat = feat_pos[:,:,:c] # [b,hw,c]
        pos = feat_pos[:,:,c:] # [b,hw,c]
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1,-2) # [b,hw,hw]
        topksim, topkid = torch.topk(sim, k=self.k, dim=-1) # [b,hw,k]
        # fetch topk feat
        topkid = topkid.flatten(1) # [b,hw*k] 
        topkfeat = feat[torch.arange(b).unsqueeze(-1), topkid] # [b,hw*k,c]
        topkfeat = topkfeat.view(b,h*w,self.k,c) # [b,hw,k,c]
        attn = topksim.softmax(dim=-1) # [b,hw,k]
        output = (attn.unsqueeze(-1) * topkfeat).sum(dim=-2) # [b,hw,c]
        output = output.view(b,h,w,c)
        output = output.permute(0,3,1,2) # [b,c,h,w]
        
        return output



class BeltramiV2(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        func = GNNV2(dim=dim, k=k)
        self.odefunc = ODEFunc(func)
        self.conv1x1 = nn.Conv2d(dim, dim*2, 1)

    def forward(self, x):
        assert len(x.shape) == 4 # [b,c,h,w]
        b,c,h,w = x.shape
        feat_pos = self.conv1x1(x) # [b,2c,h,w]

        t = torch.tensor([0,1], dtype=torch.float32, device=x.device, requires_grad=False)
        output = odeint_adjoint(self.odefunc, feat_pos, t, 
                        method=args.odeint_method,
                        rtol=args.tol, atol=args.tol)
        output = output[-1] # [b,2c,h,w]
        output = output[:,:c] # [b,c,h,w]

        return output




# ---- beltramiid

class GNNID(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        # self.fc = nn.Linear(dim, dim*2)
        self.k = k

    def forward(self, feat_pos):
        assert len(feat_pos.shape) == 4
        b,c,h,w = feat_pos.shape
        

        feat_pos = feat_pos.flatten(2) # [b,2c,hw]
        feat_pos = feat_pos.permute(0,2,1) # [b,hw,2c]
        feat = feat_pos[:,:,:c] # [b,hw,c]
        pos = feat_pos[:,:,c:] # [b,hw,c]
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1,-2) # [b,hw,hw]
        topksim, topkid = torch.topk(sim, k=self.k, dim=-1) # [b,hw,k]
        # fetch topk feat
        topkid = topkid.flatten(1) # [b,hw*k] 
        topkfeat = feat[torch.arange(b).unsqueeze(-1), topkid] # [b,hw*k,c]
        topkfeat = topkfeat.view(b,h*w,self.k,c) # [b,hw,k,c]
        attn = topksim.softmax(dim=-1) # [b,hw,k]
        output = (attn.unsqueeze(-1) * topkfeat).sum(dim=-2) # [b,hw,c]
        output = output.view(b,h,w,c)
        output = output.permute(0,3,1,2) # [b,c,h,w]
        
        return output



class BeltramiID(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        func = GNNID(dim=dim, k=k)
        self.odefunc = ODEFunc(func)
        self.conv1x1 = nn.Conv2d(dim, dim*2, 1)

    def forward(self, x):
        assert len(x.shape) == 4 # [b,c,h,w]
        b,c,h,w = x.shape
        feat_pos = self.conv1x1(x) # [b,2c,h,w]

        t = torch.tensor([0,1], dtype=torch.float32, device=x.device, requires_grad=False)
        # output = odeint_adjoint(self.odefunc, feat_pos, t, 
        #                 method=args.odeint_method,
        #                 rtol=args.tol, atol=args.tol)
        # output = output[-1] # [b,2c,h,w]
        output = self.odefunc(t, feat_pos)

        output = output[:,:c] # [b,c,h,w]

        return output






# ---- beltramiI

class GNNI(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.fc = nn.Linear(dim, dim*2)
        self.k = k
        

    def forward(self, x):
        assert len(x.shape) == 4
        b,c,h,w = x.shape
        
        x = x.flatten(2)
        x = x.permute(0,2,1) # [b,hw,c]
        feat_pos = self.fc(x) # [b,hw,c*2]
        feat = feat_pos[:,:,:c] # [b,hw,c]
        pos = feat_pos[:,:,c:] # [b,hw,c]
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1,-2) # [b,hw,hw]
        topksim, topkid = torch.topk(sim, k=self.k, dim=-1, sorted=True) # [b,hw,k]
        # fetch topk feat
        topkid = topkid.flatten(1) # [b,hw*k] 
        topkfeat = feat[torch.arange(b).unsqueeze(-1), topkid] # [b,hw*k,c]
        topkfeat = topkfeat.view(b,h*w,self.k,c) # [b,hw,k,c]
        attn = topksim.softmax(dim=-1) # [b,hw,k]
        I = torch.zeros_like(attn, requires_grad=False)
        I[:,:,0] = 1
        attn = attn - I
        x = (attn.unsqueeze(-1) * topkfeat).sum(dim=-2) # [b,hw,c]
        x = x.view(b,h,w,c)
        x = x.permute(0,3,1,2) # [b,c,h,w]
        return x
    
class BeltramiI(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        func = GNNI(dim=dim, k=k)
        self.odefunc = ODEFunc(func)

    def forward(self, x):
        t = torch.tensor([0,1], dtype=torch.float32, device=x.device, requires_grad=False)
        output = odeint_adjoint(self.odefunc, x, t, 
                        method=args.odeint_method,
                        rtol=args.tol, atol=args.tol)
        output = output[-1]
        return output







