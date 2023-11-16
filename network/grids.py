

import torch.nn as nn

import torch
from viz_lidar_mayavi_open3d import *

import MinkowskiEngine as ME

class Grids(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    








    def forward(self, cloud_feat, data_dict):


        a=1

        # -- select C
        coords = cloud_feat.C # [N,4]
        selected_indices = (coords[:,1]>0).int() *  (coords[:,2]>0).int() *  (coords[:,3]>0).int()
        selected_indices = selected_indices.bool()
        pc1 = coords[selected_indices]
        # pc1 = [pc1[pc1[:,0]==i,1:] for i in range(100) ]
        # pc1 = torch.nn.utils.rnn.pad_sequence(pc1,batch_first=True,padding_value=99999)

        # ME.spmm(cloud_feat, cloud_feat)


        pc1_feat = cloud_feat.F
        # a = cloud_feat.dense(min_coordinate=[0,0,0])

        pc1_feat = pc1_feat[selected_indices]
        pc1_feat = [pc1_feat[pc1[:,0]==i] for i in range(100) ]
        pc1_feat = torch.nn.utils.rnn.pad_sequence(pc1_feat,batch_first=True,padding_value=0)


        # # -- select decomposed coordinates
        # coords = cloud_feat.decomposed_coordinates
        # coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=99999) # [b,n,3]
        # selected_indices = (coords[:,:,0]>0).int() *  (coords[:,:,1]>0).int() *  (coords[:,:,2]>0).int()
        # selected_indices = selected_indices.bool()
        # coords = coords[selected_indices]





        return 



