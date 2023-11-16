import torch
import torch.nn as nn
from network.graph_attention_layer import GraphAttentionLayer

from network.resnetfpn_simple import ImageGeM
from torchvision.models.resnet import BasicBlock, Bottleneck


from network.gatt_block import GattBlock
from viz_lidar_mayavi_open3d import *
import torch.nn.functional as F


from tools.utils import set_seed
from tools.options import Options
args = Options().parse()
set_seed(7)








class FFB(nn.Module):

    # graph attention layer with learnable matrix

    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 n_heads=8,
                 num_blocks=3,
                 ffb_type='mix',
                 num_image_points=None,
                 num_cloud_points=None,
                 image_fe_last_dim=None
                 ):
        super(FFB, self).__init__()
        assert image_fe_last_dim is not None

        self.num_blocks = num_blocks
        self.num_image_points = num_image_points
        self.num_cloud_points = num_cloud_points


        # block_type_options = ['basicblock','bottleneck','gatt','gattm','attn','resattn','swinblock',
        #                       'qkv','qkvm','qkvg1','qkvg2','qkvmlp',None]


        window_size = [7,7]


        # assert args.gatt_fusion_block_type in block_type_options
        self.gatt_fusion_list = nn.ModuleList()
        # self.gatt_fusion_list2 = nn.ModuleList()
        # self.gatt_fusion_list3 = nn.ModuleList()

        for i in range(num_blocks):
            gatt_fusion = GattBlock(block_type=args.gatt_fusion_block_type, num_heads=n_heads, in_features=in_features)
            self.gatt_fusion_list.append(gatt_fusion)
            # gatt_fusion2 = GattBlock(block_type=args.gatt_fusion_block_type2, num_heads=n_heads, in_features=in_features)
            # self.gatt_fusion_list2.append(gatt_fusion2)
            # gatt_fusion3 = GattBlock(block_type=args.gatt_fusion_block_type3, num_heads=n_heads, in_features=in_features)
            # self.gatt_fusion_list3.append(gatt_fusion3)


        if args.use_a_fusion_first:
            gatt_fusion = GattBlock(block_type=args.gatt_fusion_block_type, num_heads=n_heads, in_features=in_features)
            self.gatt_fusion_list.append(gatt_fusion)
            # gatt_fusion2 = GattBlock(block_type=args.gatt_fusion_block_type2, num_heads=n_heads, in_features=in_features)
            # self.gatt_fusion_list2.append(gatt_fusion2)
            # gatt_fusion3 = GattBlock(block_type=args.gatt_fusion_block_type3, num_heads=n_heads, in_features=in_features)
            # self.gatt_fusion_list3.append(gatt_fusion3)



        assert ffb_type in ['mix','pure','imageonly']
        # if ffb_type=='mix':
        # assert args.gatt_image_block_type in block_type_options
        # assert args.gatt_cloud_block_type in block_type_options
        self.gatt_image_list = nn.ModuleList()
        self.gatt_cloud_list = nn.ModuleList()


        for i in range(num_blocks):
            gatt_image = GattBlock(block_type=args.gatt_image_block_type, num_heads=n_heads, in_features=in_features, 
                                   window_size=[7,7], shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size])
            gatt_cloud = GattBlock(block_type=args.gatt_cloud_block_type, num_heads=n_heads, in_features=in_features)

            self.gatt_image_list.append(gatt_image)
            self.gatt_cloud_list.append(gatt_cloud)


        self.fc128_256 = nn.Linear(128, image_fe_last_dim)
        self.fc128_256_image = nn.Linear(128, image_fe_last_dim)


        self.fusion_gem_128 = ImageGeM()
        self.fusion_gem_256 = ImageGeM()





    def forward(self, in_image_feat, in_cloud_feat, in_image_feat_gem=None, in_cloud_feat_gem=None, in_data_dict=None):

        assert len(in_image_feat.shape)==3
        assert len(in_cloud_feat.shape)==3

        b = in_image_feat.shape[0]
        # h, w = in_image_feat.shape[-2:]
        # -- flatten image feat
        # in_image_feat = in_image_feat.flatten(2).permute(0,2,1) # [b,hw,c]
        num_image_points = in_image_feat.shape[1]
        num_cloud_points = in_cloud_feat.shape[1]


        if args.use_l2norm_before_fusion:
            in_image_feat = F.normalize(in_image_feat, p=2, dim=-1)
            in_cloud_feat = F.normalize(in_cloud_feat, p=2, dim=-1)
        




        # x:[image, cloud]  [b,num_pts,c]  image(hw)
        assert args.ffb_type in ['mix','pure','imageonly']

        
        fusion_feat = torch.cat([in_image_feat, in_cloud_feat], dim=1)

        # _x = fusion_feat
        if args.use_a_fusion_first:
            fusion_feat = self.gatt_fusion_list[0](fusion_feat)
            # if args.gatt_fusion_block_type2 is not None:
            #     fusion_feat2 = self.gatt_fusion_list2[0](_x)
            #     fusion_feat = fusion_feat + fusion_feat2
            # if args.gatt_fusion_block_type3 is not None:
            #     fusion_feat3 = self.gatt_fusion_list3[0](_x)
            #     fusion_feat = fusion_feat + fusion_feat3




        # _x = fusion_feat
        for i in range(self.num_blocks):
            fusion_feat = self.gatt_fusion_list[i+1](fusion_feat)
            # if args.gatt_fusion_block_type2 is not None:
            #     fusion_feat2 = self.gatt_fusion_list2[i+1](_x)
            #     fusion_feat = fusion_feat + fusion_feat2
            # if args.gatt_fusion_block_type3 is not None:
            #     fusion_feat3 = self.gatt_fusion_list3[i+1](_x)
            #     fusion_feat = fusion_feat + fusion_feat3





        output_feat_gem_128 = self.fusion_gem_128(fusion_feat.permute(0,2,1).unsqueeze(-1))





        assert args.make_image_fusion_same_channel in ['fconfusion','fconfusiongem']
        if args.make_image_fusion_same_channel == 'fconfusion':
            output_feat_gem_256 = self.fusion_gem_256(self.fc128_256(fusion_feat).permute(0,2,1).unsqueeze(-1))
        elif args.make_image_fusion_same_channel == 'fconfusiongem':
            output_feat_gem_256 = self.fc128_256(output_feat_gem_128)





        output_image_feat_128 = fusion_feat[:,:num_image_points] # [100,300,128]
        output_cloud_feat = fusion_feat[:,num_image_points:num_image_points + num_cloud_points] # [100,128,128]


        output_image_feat_256 = self.fc128_256_image(output_image_feat_128)


        output_dict = {
            'output_image_feat_128':output_image_feat_128,
            'output_image_feat_256':output_image_feat_256,
            'output_cloud_feat':output_cloud_feat,
            'output_fusion_feat':fusion_feat,
            'output_feat_gem_128':output_feat_gem_128,
            'output_feat_gem_256':output_feat_gem_256,

        }


        return output_dict
    


