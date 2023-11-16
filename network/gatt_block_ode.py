



# from network.gatt_block import (
#     QKVAttention,
#     QKVAttentionM,
#     QKVAttentionG1,
#     QKVAttentionG2,

#     XQKVAttention,
#     XQKVAttentionG1,
#     XQKVAttentionG2,
#     XQKVAttentionM,
# )

from torchdiffeq import odeint_adjoint, odeint
import torch
import torch.nn as nn


from tools.options import Options
args = Options().parse()




class ODEMultiheadAttention(nn.Module):
    """
    change vanilla attn function into ODE function
    """
    def __init__(self, attnfunc):
        super().__init__()
        self.attnfunc = attnfunc

    def forward(self, t, x):
        if isinstance(self.attnfunc, nn.MultiheadAttention):
            output, _ = self.attnfunc(x,x,x)
        else:
            output = self.attnfunc(x)
        # print(t)
        return output
    




class ODEInt(nn.Module):
    """  
    conduct ODE integration
    """
    def __init__(self, odefunc, tol=1e-3) -> None:
        super().__init__()
        self.tol = tol
        self.odefunc = odefunc

    def forward(self, x):
        output = odeint_adjoint(self.odefunc, y0=x, t=torch.tensor([0.,1.]).to(device=x.device), 
                                # method='dopri5',
                                method=args.odeint_method,
                                atol=self.tol, 
                                rtol=self.tol)[-1]
        return output
    
    


