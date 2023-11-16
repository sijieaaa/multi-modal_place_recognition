import torch
import torch.nn as nn
from network.graph_attention_layer import GraphAttentionLayer

from network.resnetfpn_simple import ImageGeM
from torchvision.models.resnet import BasicBlock, Bottleneck


from network.gatt_block import GattBlock
from viz_lidar_mayavi_open3d import *
import torch.nn.functional as F

import MinkowskiEngine as ME

import spconv
import torch.nn.functional as F

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from MinkowskiEngine import SparseTensor

from network.gatt_block import GattBlock

from tools.utils import set_seed
from tools.options import Options
args = Options().parse()
set_seed(7)











class LaterImageBranch(nn.Module):


    def __init__(self, 
                 block_type, 
                 num_heads, 
                 in_features, 
                 num_blocks=1,
                 window_size=None, 
                 shift_size=None):
        super(LaterImageBranch, self).__init__()

        self.num_blocks = num_blocks


        self.seq = nn.ModuleList()
        for i in range(num_blocks):
            block = GattBlock(block_type=args.later_image_branch_type,
                            num_heads=num_heads,
                            in_features=in_features,
                            window_size=window_size,
                            shift_size=shift_size)
            self.seq.append(block)


        
        self.conv1x1 = nn.Conv2d(in_features, 128, kernel_size=1)
        self.imagegem = ImageGeM()

        self.w = nn.Parameter(torch.ones([1],dtype=torch.float32),requires_grad=True)



    def forward(self, image_feat, other_feat=None):
        
        assert len(image_feat.shape)==4
        b,c,h,w = image_feat.shape 

        # -- flatten image
        image_feat = torch.flatten(image_feat, start_dim=2)
        image_feat = image_feat.permute(0,2,1) # [b,hw,c]




        if args.later_image_branch_interaction_type is not None:
            if 'sigmoid' in args.later_image_branch_interaction_type:
                other_feat = F.sigmoid(other_feat)
            if 'w' in args.later_image_branch_interaction_type:
                other_feat = other_feat * self.w
            if 'times' in args.later_image_branch_interaction_type:
                image_feat = image_feat * other_feat.unsqueeze(1)
                # image_feat = image_feat * other_feat
            elif 'add' in args.later_image_branch_interaction_type:
                image_feat = image_feat + other_feat.unsqueeze(1)
                # image_feat = image_feat + other_feat








        for i in range(self.num_blocks):
            image_feat = self.seq[i](image_feat,h,w)




        # -- flatten back
        image_feat = image_feat.permute(0,2,1)
        image_feat = image_feat.view(b,c,h,w)
        image_feat_256 = image_feat


        image_feat_128 = self.conv1x1(image_feat)


        image_feat_256_gem = self.imagegem(image_feat)


        return image_feat_128, image_feat_256_gem, image_feat_256
    


