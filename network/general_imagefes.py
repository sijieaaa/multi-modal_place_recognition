# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images

import torch
import torch.nn as nn
import torchvision.models as TVmodels
# from TV_offline_models.swin_transformer import swin_v2_t,swin_v2_s
import MinkowskiEngine as ME

from models.minkloc import MinkLoc
from network.resnetfpn_simple import ImageGeM

from tools.utils import set_seed
set_seed(7)
from tools.options import Options
# args = Options().parse()





class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)








class GeneralImageFE(torch.nn.Module):
    def __init__(self, 
                 image_fe,
                 num_other_stage_blocks,
                 num_stage3_blocks,
                 input_type,
                 out_channels: int=None, lateral_dim: int=None, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='gem',
                 image_fe_dim=None):
        super().__init__()
        '''
        resnet [64,64,128,256,512]
        convnext [96,96,192,384,768]
        swin [96,96,192,384,768]
        swin_v2 [96,96,192,384,768]
        '''

        assert input_type in ['image','sph_cloud']

        self.image_fe = image_fe
        self.num_other_stage_blocks = num_other_stage_blocks
        self.num_stage3_blocks = num_stage3_blocks
        self.input_type = input_type


        self.out_channels = out_channels
        # self.fh_num_bottom_up = fh_num_bottom_up
        # self.fh_num_top_down = fh_num_top_down
        # self.lateral_dim = lateral_dim
        self.pool_method = pool_method
        # self.add_fc_block = add_fc_block

        # -- resnet
        if self.image_fe == 'resnet18':
            self.model = TVmodels.resnet18(weights='IMAGENET1K_V1')
            self.last_dim = 256
        elif self.image_fe == 'resnet34':
            self.model = TVmodels.resnet34(weights='IMAGENET1K_V1')
            self.last_dim = 256
        elif self.image_fe == 'resnet50':
            self.model = TVmodels.resnet50(weights='IMAGENET1K_V2')
            self.last_dim = 1024
        elif self.image_fe == 'resnet101':
            self.model = TVmodels.resnet101(weights='IMAGENET1K_V2')
            self.last_dim = 1024
        elif self.image_fe == 'resnet152':
            self.model = TVmodels.resnet152(weights='IMAGENET1K_V2')
            self.last_dim = 1024

        # -- convnext
        elif self.image_fe == 'convnext_tiny':
            self.model = TVmodels.convnext_tiny(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'convnext_small':
            self.model = TVmodels.convnext_small(weights='IMAGENET1K_V1')
            self.last_dim = 384

        # -- swin
        elif self.image_fe == 'swin_t':
            self.model = TVmodels.swin_t(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_s':
            self.model = TVmodels.swin_s(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_t':
            self.model = TVmodels.swin_v2_t(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_s':
            self.model = TVmodels.swin_v2_s(weights='IMAGENET1K_V1')
            self.last_dim = 384

        # -- efficientnet
        elif self.image_fe == 'efficientnet_b0':
            self.model = TVmodels.efficientnet_b0(weights='IMAGENET1K_V1')
            self.last_dim = 112
        elif self.image_fe == 'efficientnet_b1':
            self.model = TVmodels.efficientnet_b1(weights='IMAGENET1K_V2')
            self.last_dim = 112
        elif self.image_fe == 'efficientnet_b2':
            self.model = TVmodels.efficientnet_b2(weights='IMAGENET1K_V1')
            self.last_dim = 120
        elif self.image_fe == 'efficientnet_v2_s':
            self.model = TVmodels.efficientnet_v2_s(weights='IMAGENET1K_V1')
            self.last_dim = 160

        # -- regnet
        elif self.image_fe == 'regnet_x_3_2gf':
            self.model = TVmodels.regnet_x_3_2gf(weights='IMAGENET1K_V2')
            self.last_dim = 432
        elif self.image_fe == 'regnet_y_1_6gf':
            self.model = TVmodels.regnet_y_1_6gf(weights='IMAGENET1K_V2')
            self.last_dim = 336
        elif self.image_fe == 'regnet_y_3_2gf':
            self.model = TVmodels.regnet_y_3_2gf(weights='IMAGENET1K_V2')
            self.last_dim = 576




        self.conv1x1 = nn.Conv2d(self.last_dim, image_fe_dim, kernel_size=1)

        self.image_gem = ImageGeM()



    def forward_resnet(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x = self.model.layer4(x)

        return x



    def forward_convnext(self, x):
        layers_list = list(self.model.features.children())
        assert len(layers_list)==8
        layers_list[1] = layers_list[1][:self.num_other_stage_blocks]
        layers_list[3] = layers_list[3][:self.num_other_stage_blocks]
        layers_list[5] = layers_list[5][:self.num_stage3_blocks]
        layers_list = layers_list[:-2]
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x

    
    def forward_swin(self, x):
        layers_list = list(self.model.features.children())
        layers_list = layers_list[:-2]
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        x = x.permute(0,3,1,2)
        return x



    def forward_efficientnet(self, x):
        # x = self.model.features(x)
        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.classifier(x)
        # b0[:-3]112  b1[:-3]112  b2[:-3]120  v2_s[:-2]160
        layers_list = list(self.model.features.children())
        if self.image_fe == 'efficientnet_v2_s':
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list[:-3]
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x



    def forward_regnet(self, x):
        # x = self.model.stem(x)
        # x = self.model.trunk_output(x)
        # x = self.model.avgpool(x)
        # x = x.flatten(start_dim=1)
        # x = self.model.fc(x)
        # regnet_x_3_2gf[:-1]432  regnet_y_1_6gf[:-1]336  regnet_y_3_2gf[:-1]576
        x = self.model.stem(x)
        layers_list = list(self.model.trunk_output.children())
        layers_list = layers_list[:-1]
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x




    def forward(self, data_dict):
        if self.input_type == 'image':
            x = data_dict['images']
        elif self.input_type == 'sph_cloud':
            x = data_dict['sph_cloud']


        
        if self.image_fe in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            x = self.forward_resnet(x)
        elif self.image_fe in ['convnext_tiny','convnext_small']:
            x = self.forward_convnext(x)
        elif self.image_fe in ['swin_t','swin_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['swin_v2_t','swin_v2_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_v2_s']:
            x = self.forward_efficientnet(x)
        elif self.image_fe in ['regnet_x_3_2gf','regnet_y_1_6gf','regnet_y_3_2gf']:
            x = self.forward_regnet(x)
        else:
            raise NotImplementedError(f'not supported {self.image_fe}')
        

        x_feat_256 = x

        x_gem_256 = self.image_gem(x_feat_256)

        x = self.conv1x1(x)


        x_gem = self.image_gem(x)


        return x, x_gem, x_feat_256, x_gem_256
    




