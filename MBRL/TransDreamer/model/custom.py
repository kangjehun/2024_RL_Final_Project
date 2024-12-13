import torch.nn as nn

class Conv2DBlock(nn.Module):
    
    def __init__(self, c_in, c_out, k, s, p, bias=True,
                 num_groups=0, weight_init='xavier',
                 non_linearity=True, act='elu'):
        
        super().__init__()
        
        self.net = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        
        # Weight initialization
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(self.net.weight)
        else:
            nn.init.kaiming_uniform_(self.net.weight)
        
        # Bias initialization
        if bias:
            nn.init.zeros_(self.net.bias)
        
        # Group normalization
        if num_groups > 0: 
            self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
        
        # Non-linear activation
        if non_linearity:
            if act == 'relu':
                self.non_linear = nn.ReLU()
            elif act == 'elu':
                self.non_linear = nn.ELU()
            else:
                self.non_linear = nn.CELU()
        
        self.non_linearity = non_linearity
        self.num_groups = num_groups

    def forward(self, inputs):
        
        output = self.net(inputs)
        
        # Group normalization
        if self.num_groups > 0:
            output = self.group_norm(output)
            
        # Non-linear activation
        if self.non_linearity:
            output = self.non_linear(output)
            
        return output

class Linear(nn.Module):
    
    def __init__(self, dim_in, dim_out, bias=True, weight_init='xavier'):
        
        super().__init__()
        
        self.net = nn.Linear(dim_in, dim_out, bias=bias)
        
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(self.net.weight)
        else:
            nn.init.kaiming_uniform_(self.net.weight)
        
        if bias:
            nn.init.zeros_(self.net.bias)
    
    def forward(self, inputs):
        return self.net(inputs)

class MLP(nn.Module):
    
    def __init__(self, dims, act, weight_init='xavier', output_act=None, norm=False):
        
        super().__init__()
        
        dims_in = dims[:-2]
        dims_out = dims[1:-1]
        
        layers = []
        for d_in, d_out in zip(dims_in, dims_out):
            layers.append(Linear(d_in, d_out))
            if norm:
                layers.append(nn.LayerNorm(d_out))
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.CELU())
        
        layers.append(Linear(d_out, dims[-1]))
        if output_act:
            if norm:
                layers.append(nn.LayerNorm)
            if act == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.CELU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
        