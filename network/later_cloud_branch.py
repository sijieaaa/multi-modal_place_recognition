import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock



from viz_lidar_mayavi_open3d import *

import torch.nn.functional as F

import MinkowskiEngine as ME
from layers.pooling import MinkGeM as MinkGeM
from layers.pooling import MinkSpconvGeM



import spconv.pytorch as spconv
import torch.nn.functional as F

from MinkowskiEngine.modules.resnet_block import BasicBlock
from tools.utils import set_seed
from tools.options import Options
args = Options().parse()
set_seed(7)







class BasicBlock(nn.Module):


    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class SubMBasicBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(SubMBasicBlock, self).__init__()

        self.conv1 = spconv.SubMConv3d(inplanes,planes,kernel_size=3,stride=1,padding=1)
        self.norm1 = nn.BatchNorm1d(planes)
        self.conv2 = spconv.SubMConv3d(inplanes,planes,kernel_size=3,stride=1,padding=1)
        self.norm2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.norm1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))

        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))

        return out



def spconv_add_mul(sparseconvtensor, mat, operation=None):
    assert isinstance(sparseconvtensor, spconv.SparseConvTensor)
    assert operation in ['add','mul']
    features = sparseconvtensor.features
    indices = sparseconvtensor.indices
    batch_size = sparseconvtensor.batch_size
    for i in range(batch_size):
        row_ids = indices[:,0]==i
        if operation=='add':
            features[row_ids] = features[row_ids] + mat[i:i+1]
        if operation=='mul':
            features[row_ids] = features[row_ids] * mat[i:i+1]

    sparseconvtensor = sparseconvtensor.replace_feature(feature=features)
    
    return sparseconvtensor






class LaterCloudBranch(nn.Module):


    def __init__(self, 
                 in_features, 
                 num_blocks=1,
                 D=3
                 ):
        super(LaterCloudBranch, self).__init__()




        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()
        self.broadcast_add = ME.MinkowskiBroadcastAddition()


        self.seq = nn.ModuleList()
        # for i in range(num_blocks-1):
        for i in range(num_blocks):

            if args.later_cloud_branch_type == 'basicblock':
                block = BasicBlock(inplanes=in_features,
                                    planes=in_features,
                                    stride=1,
                                    downsample=None,
                                    dimension=3)
            elif args.later_cloud_branch_type == 'submbasicblock':
                block = SubMBasicBlock(inplanes=in_features,
                                    planes=in_features,
                                    stride=1,
                                    downsample=None)
            else: 
                raise Exception

            self.seq.append(block)
        self.seq = nn.Sequential(*self.seq)




        self.minkspconvgem = MinkSpconvGeM(input_dim=in_features)

        self.w = nn.Parameter(torch.ones([1],dtype=torch.float32),requires_grad=True)
        a=1
     


    def forward(self, cloud_feat, other_feat=None):


        if args.later_cloud_branch_interaction_type is not None:
            if 'sigmoid' in args.later_cloud_branch_interaction_type:
                other_feat = F.sigmoid(other_feat)
            if 'w' in args.later_cloud_branch_interaction_type:
                other_feat = other_feat * self.w

            
            if isinstance(cloud_feat, ME.SparseTensor):
                _pool_feat = self.avg_pool(cloud_feat)
                other_feat = ME.SparseTensor(other_feat, 
                                            coordinate_manager=_pool_feat.coordinate_manager,
                                            coordinate_map_key=_pool_feat.coordinate_map_key)
                if 'times' in args.later_cloud_branch_interaction_type:
                    cloud_feat = self.broadcast_mul(cloud_feat, other_feat)
                elif 'add' in args.later_cloud_branch_interaction_type:
                    cloud_feat = self.broadcast_add(cloud_feat, other_feat)

            elif isinstance(cloud_feat, spconv.SparseConvTensor):
                if 'times' in args.later_cloud_branch_interaction_type:
                    cloud_feat = spconv_add_mul(cloud_feat, other_feat, 'mul')
                elif 'add' in args.later_cloud_branch_interaction_type:
                    cloud_feat = spconv_add_mul(cloud_feat, other_feat, 'add')





        cloud_feat = self.seq(cloud_feat)

        # cloud_feat_gem = self.minkgem(cloud_feat)
        cloud_feat_gem = self.minkspconvgem(cloud_feat)


        return cloud_feat, cloud_feat_gem
    


