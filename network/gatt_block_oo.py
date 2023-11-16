


import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import torch

from tools.options import Options
args = Options().parse()




class OOQKVAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, act_type, ooqkv_type):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.ooqkv_type = ooqkv_type

        if act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'id':
            self.act = nn.Identity()
        else:
            raise NotImplementedError


    def func_q(self, t, x):
        output = self.q(x)
        output = self.act(output)
        return output
    
    def func_k(self, t, x):
        output = self.k(x)
        output = self.act(output)
        return output
    
    def func_v(self, t, x):
        output = self.v(x)
        output = self.act(output)
        return output


    def forward_ode_func(self, x, func):
        t = torch.tensor([0,1]).float().to(x.device)
        if args.odeint_method == 'id':
            output = func(t, x)
        else:
            output = odeint(func=func, y0=x, t=t, 
                                    method=args.odeint_method,
                                    atol=args.tol,
                                    rtol=args.tol,
                                    )[-1]
        return output
    

        
    def forward(self, x):
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        if self.ooqkv_type == 'q':
            q = self.forward_ode_func(x, self.func_q)
            k = self.k(x)
            v = self.v(x)
        if self.ooqkv_type == 'k':
            k = self.forward_ode_func(x, self.func_k)
            q = self.q(x)
            v = self.v(x)
        if self.ooqkv_type == 'v':
            v = self.forward_ode_func(x, self.func_v)
            q = self.q(x)
            k = self.k(x)


        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)

        q = q.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]

        att = q @ k.permute(0,1,3,2) # [batch, num_heads, seq_len, seq_len]

        att = F.softmax(att, dim=-1)

        output = att @ v # [batch, num_heads, seq_len, head_dim]
        output = output.permute(0,2,1,3) # [batch, seq_len, num_heads, head_dim]
        output = output.contiguous()
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output








class OOQKVAttentionG1(nn.Module):
    def __init__(self, embed_dim, num_heads, act_type, ooqkv_type) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.g = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads 
        self.ooqkv_type = ooqkv_type
        
        if act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'id':
            self.act = nn.Identity()
        else:
            raise NotImplementedError


    def func_g(self, t, x):
        output = self.g(x)
        output = self.act(output)
        return output


    def forward_ode_func(self, x, func):
        t = torch.tensor([0,1]).float().to(x.device)
        if args.odeint_method == 'id':
            output = func(t, x)
        else:
            output = odeint(func=func, y0=x, t=t, 
                                    method=args.odeint_method,
                                    atol=args.tol,
                                    rtol=args.tol,
                                    )[-1]
        return output
    

    def forward(self, x):
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        if 'g1' in self.ooqkv_type:
            g = self.forward_ode_func(x, self.func_g)
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
        else:
            raise NotImplementedError

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        g = g.view(g.shape[0], g.shape[1], self.num_heads, -1)

        q = q.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        g = g.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]

        qg = q * g
        if 'act' in self.ooqkv_type:
            qg = self.act(qg)
        att = qg @ k.permute(0,1,3,2)
        att = F.softmax(att, dim=-1)

        output = att @ v # [batch, num_heads, seq_len, head_dim]
        output = output.permute(0,2,1,3) # [batch, seq_len, num_heads, head_dim]
        output = output.contiguous()
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output








class OOQKVAttentionG1G(nn.Module):
    def __init__(self, embed_dim, num_heads, act_type, ooqkv_type) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.gin = nn.Linear(embed_dim, embed_dim)
        self.gout = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads 
        self.ooqkv_type = ooqkv_type
        
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'id':
            self.act = nn.Identity()
        else:
            raise NotImplementedError


    def func_g(self, t, x):
        output = self.gin(x)
        output = self.act(output)
        output = self.gout(output)
        return output


    def forward_ode_func(self, x, func):
        t = torch.tensor([0,1]).float().to(x.device)
        if args.odeint_method == 'id':
            output = func(t, x)
        else:
            output = odeint(func=func, y0=x, t=t, 
                                    method=args.odeint_method,
                                    atol=args.tol,
                                    rtol=args.tol,
                                    )[-1]
        return output
    

    def forward(self, x):
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        if 'g1g' in self.ooqkv_type:
            g = self.forward_ode_func(x, self.func_g)
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
        else:
            raise NotImplementedError

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        g = g.view(g.shape[0], g.shape[1], self.num_heads, -1)

        q = q.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        g = g.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]

        qg = q * g
        if 'act' in self.ooqkv_type:
            qg = self.act(qg)
        att = qg @ k.permute(0,1,3,2)
        att = F.softmax(att, dim=-1)

        output = att @ v # [batch, num_heads, seq_len, head_dim]
        output = output.permute(0,2,1,3) # [batch, seq_len, num_heads, head_dim]
        output = output.contiguous()
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output








class OOQKVAttentionG2(nn.Module):
    def __init__(self, embed_dim, num_heads, act_type, ooqkv_type) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads 
        self.head_dim = head_dim
        self.ooqkv_type = ooqkv_type

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.g = nn.Linear(embed_dim, num_heads * head_dim * head_dim)

        if act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'id':
            self.act = nn.Identity()
        else:
            raise NotImplementedError


    def func_g(self, t, x):
        output = self.g(x)
        output = self.act(output)
        return output


    def forward_ode_func(self, x, func):
        t = torch.tensor([0,1]).float().to(x.device)
        if args.odeint_method == 'id':
            output = func(t, x)
        else:
            output = odeint(func=func, y0=x, t=t, 
                                    method=args.odeint_method,
                                    atol=args.tol,
                                    rtol=args.tol,
                                    )[-1]
        return output
    

    def forward(self, x):
        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        if 'g2' in self.ooqkv_type:
            g = self.forward_ode_func(x, self.func_g)
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
        else:
            raise NotImplementedError

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        g = g.view(g.shape[0], g.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim*head_dim]
        g = g.view(g.shape[0], g.shape[1], self.num_heads, self.head_dim, self.head_dim) # [batch, seq_len, num_heads, head_dim, head_dim]

        q = q.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        v = v.permute(0,2,1,3) # [batch, num_heads, seq_len, head_dim]
        g = g.permute(0,2,1,3,4) # [batch, num_heads, seq_len, head_dim, head_dim]


        q = q.unsqueeze(-2) # [batch, num_heads, seq_len, 1, head_dim]

        att = q @ g # [batch, num_heads, seq_len, 1, head_dim]
        att = att.squeeze(-2) # [batch, num_heads, seq_len, head_dim]
        att = att @ k.permute(0,1,3,2)  # [batch, num_heads, seq_len, seq_len]

        att = F.softmax(att, dim=-1) # [batch, num_heads, seq_len, seq_len]

        output = att @ v # [batch, num_heads, seq_len, head_dim]
        output = output.permute(0,2,1,3) # [batch, seq_len, num_heads, head_dim]
        output = output.contiguous()
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output
        