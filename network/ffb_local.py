import torch
import torch.nn as nn
from network.graph_attention_layer import GraphAttentionLayer

from network.resnetfpn_simple import ImageGeM
from torchvision.models.resnet import BasicBlock, Bottleneck
import MinkowskiEngine as ME

from network.swinblock import ShiftedWindowAttention

from network.gatt_block import GattBlock
from viz_lidar_mayavi_open3d import *
# from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from third_party.SparseTransformer import sptr

from datasets.oxford import transform_pts_in_camsystem, project_onto_image

from datasets.oxford import viz_pts_projected_on_image

from torchdiffeq import odeint_adjoint, odeint

from network.beltrami import Beltrami
from network.beltrami import BeltramiV2
from network.beltrami import BeltramiID
from network.beltrami import BeltramiI

import matplotlib.pyplot as plt

from torch_geometric.nn import voxel_grid

from tools.utils import set_seed
from tools.options import Options
args = Options().parse()
set_seed(7)






def convert_pts_to_uv(cloud_feat, T_camera_lidar_basedon_pose, P0_camera):
    '''
    pts:     Tensor [N,4] 
    pts_list: List [Ni,3] 
    T_camera_lidar_basedon_pose: [4,4]
    P0_camera: [4,4]
    '''
    assert isinstance(cloud_feat, ME.SparseTensor)
    pts = cloud_feat.C
    pts_list = cloud_feat.decomposed_coordinates
    feats = cloud_feat.F
    assert pts.shape[-1] == 4
    assert isinstance(pts, torch.Tensor)
    assert isinstance(pts_list, list)
    assert isinstance(T_camera_lidar_basedon_pose, torch.Tensor)
    assert isinstance(P0_camera, torch.Tensor)

    if args.dataset=='boreas':
        selected_indices = [(e[:,0]>0) & (e[:,0]<1000) \
                & (e[:,1]>0) & (e[:,1]<1000) \
                & (e[:,2]>-10) & (e[:,2]<50) for e in pts_list]

    pts_in_camsystem = [transform_pts_in_camsystem(e.float(), eT) for e, eT in zip(pts_list, T_camera_lidar_basedon_pose)]
    # uv, colors, mask = project_onto_image(pts_in_camsystem, P0_camera)
    # a = [project_onto_image(e, eP) for e, eP in zip(pts_in_camsystem, P0_camera)]
    uvcolorsmask = [project_onto_image(e, eP) for e, eP in zip(pts_in_camsystem, P0_camera)]
    uv = [e[0] for e in uvcolorsmask]
    colors = [e[1] for e in uvcolorsmask]
    mask = [e[2] for e in uvcolorsmask]
    maskselect = [emask & eselect for emask, eselect in zip(mask, selected_indices)]



    maskselect = torch.cat(maskselect, dim=0)
    uv = torch.cat(uv, dim=0)
    colors = torch.cat(colors, dim=0)


    pts_in_image = pts[maskselect]
    pts_uv_in_image = uv[maskselect]
    pts_colors_in_image = colors[maskselect]
    pts_out_image = pts[~maskselect]
    feats_in_image = feats[maskselect]
    feats_out_image = feats[~maskselect]

    pts_uv_in_image = torch.cat([pts_in_image[:,0:1], pts_uv_in_image], dim=-1)


    assert pts_in_image.shape[0] == pts_uv_in_image.shape[0]
    assert pts_in_image.shape[0] == pts_colors_in_image.shape[0]
    assert pts_in_image.shape[0] == feats_in_image.shape[0]
    assert len(pts_in_image)+len(pts_out_image) == len(pts)
    assert len(feats_in_image)+len(feats_out_image) == len(pts)

    return pts_in_image, pts_uv_in_image, pts_colors_in_image, pts_out_image, feats_in_image, feats_out_image



def convert_image_to_uv(image_feat):
    '''
    image_feat: [b,c,h,w]
    '''

    b,c,h,w = image_feat.shape
    image_pixel_coords_wh = make_coord_grid(h,w) # [h,w,2]
    image_pixel_coords_wh = image_pixel_coords_wh.unsqueeze(0).repeat(b,1,1,1) # [b,h,w,2]
    batch_ids = torch.arange(0,b).view(-1,1,1,1).repeat(1,h,w,1) # [b,h,w,1]
    image_pixel_coords_wh = torch.cat([batch_ids, image_pixel_coords_wh], dim=-1) # [b,h,w,3]
    image_pixel_coords_wh = image_pixel_coords_wh.view(b,-1,3) # [b,h*w,3]  
    image_pixel_coords_wh = image_pixel_coords_wh.view(-1,3) # [b*h*w,3]


    image_feat = image_feat.permute(0,2,3,1) # [b,h,w,c]
    image_feat = image_feat.view(b,-1,c) # [b,h*w,c]
    image_feat = image_feat.contiguous()
    image_feat = image_feat.view(-1,c) # [b*h*w,c]


    return image_pixel_coords_wh, image_feat



def make_coord_grid(h, w):
    '''
    h: int
    w: int
    '''
    image_pixel_coords_h = torch.arange(0,h).view(-1,1).repeat(1,w)
    image_pixel_coords_w = torch.arange(0,w).view(1,-1).repeat(h,1)
    image_pixel_coords_wh = torch.stack([image_pixel_coords_w, image_pixel_coords_h], dim=-1) # [h,w,2]
    
    # image_pixel_coords_wh = image_pixel_coords_wh.view(-1,2) # [h*w,2]
    return image_pixel_coords_wh






class ODEFunc(nn.Module):
    def __init__(self, func):
        super(ODEFunc, self).__init__()
        self.func = func
        # self.iter = 0
    def forward(self, t, x):
        x = self.func(x)
        x = F.relu(x, inplace=True)
        # self.iter += 1
        return x




class ODEFuncWORelu(nn.Module):
    def __init__(self, func, use_relu=False):
        super(ODEFuncWORelu, self).__init__()
        self.func = func
        self.use_relu = use_relu
        # self.iter = 0
    def forward(self, t, x):
        x = self.func(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        # self.iter += 1
        return x





class FFBLocal(nn.Module):
    '''
    '''
    def __init__(self, 
                 in_image_feat_dim,
                 embed_dim, 
                 num_heads=8, 
                 num_blocks=1,
                 block_type='attn',
                 window_size=None, 
                 shift_win=False
                 ):
        super(FFBLocal, self).__init__()
        assert window_size is not None
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'


        self.in_image_feat_dim = in_image_feat_dim
        self.embed_dim = embed_dim

        self.attn = sptr.VarLengthMultiheadSA(embed_dim=embed_dim, num_heads=num_heads, indice_key='None', 
                                         window_size=window_size, shift_win=shift_win,
                                         attntion_type=args.ffblocal_block_type,
                                         act_type=args.gatt_fusion_act_type
                                         )
        # self.attn2 = sptr.VarLengthMultiheadSA(embed_dim=embed_dim, num_heads=num_heads, indice_key='None', 
        #                                  window_size=window_size, shift_win=shift_win,
        #                                  attntion_type=args.gatt_fusion_block_type2
        #                                  )
        # self.attn3 = sptr.VarLengthMultiheadSA(embed_dim=embed_dim, num_heads=num_heads, indice_key='None', 
        #                                  window_size=window_size, shift_win=shift_win,
        #                                  attntion_type=args.gatt_fusion_block_type3
        #                                  )


        self.conv256_128 = nn.Conv2d(in_image_feat_dim, embed_dim, 1)
        self.conv128_256 = nn.Conv2d(embed_dim, in_image_feat_dim, 1)
        self.inimagefc128_128 = nn.Linear(embed_dim, embed_dim)
        self.inimagefc128_128_ode = ODEFunc(nn.Linear(embed_dim, embed_dim))
        self.outimagefc128_128 = nn.Linear(embed_dim, embed_dim)
        self.outimagefc128_128_ode = ODEFunc(nn.Linear(embed_dim, embed_dim))
        




        self.inconvme128_128 = ME.MinkowskiConvolution(embed_dim, embed_dim, kernel_size=1, dimension=3)
        self.outconvme128_128 = ME.MinkowskiConvolution(embed_dim, embed_dim, kernel_size=1, dimension=3)


        self.infc128_128 = nn.Linear(embed_dim, embed_dim)
        self.outfc128_128 = nn.Linear(embed_dim, embed_dim)

        self.infc128_128_ode = ODEFunc(nn.Linear(embed_dim, embed_dim))
        self.infc64_64_ode = ODEFunc(nn.Linear(embed_dim//2, embed_dim//2))
        self.infc128_64 = nn.Linear(embed_dim, embed_dim//2)
        self.infc64_128 = nn.Linear(embed_dim//2, embed_dim)

        self.outfc128_128_ode = ODEFunc(nn.Linear(embed_dim, embed_dim))
        self.outfc64_64_ode = ODEFunc(nn.Linear(embed_dim//2, embed_dim//2))
        self.outfc128_64 = nn.Linear(embed_dim, embed_dim//2)
        self.outfc64_128 = nn.Linear(embed_dim//2, embed_dim)


        self.swinlayer = ShiftedWindowAttention(dim=embed_dim, window_size=[args.imageswin_windowsize, args.imageswin_windowsize], 
                                                shift_size=[0,0], num_heads=8)
        self.swinlayer_ode = ODEFuncWORelu(ShiftedWindowAttention(
                                                            dim=embed_dim, window_size=[args.imageswin_windowsize, args.imageswin_windowsize], 
                                                            shift_size=[0,0], num_heads=8, use_proj=args.imageswin_useproj),
                                            use_relu=args.imageswin_userelu)

        if 'beltrami' in args.imageswin_type:
            beltrami_type = args.imageswin_type.split('_')[0]
            beltrami_k = args.imageswin_type.split('_')[1]
            assert 'k' in beltrami_k
            beltrami_k = int(beltrami_k[1:])
            if beltrami_type == 'beltrami':
                self.beltrami = Beltrami(dim=embed_dim, k=beltrami_k)
            elif beltrami_type == 'beltramiv2':
                self.beltrami = BeltramiV2(dim=embed_dim, k=beltrami_k)
            elif beltrami_type == 'beltramiid':
                self.beltrami = BeltramiID(dim=embed_dim, k=beltrami_k)
            elif beltrami_type == 'beltramiI':
                self.beltrami = BeltramiI(dim=embed_dim, k=beltrami_k)
            else:
                raise NotImplementedError


        self.tol = 1e-2
        
        self.t = nn.Parameter(torch.tensor([0,1]).float(), requires_grad=False)






    def forward(self, in_image_feat, in_cloud_feat, T_camera_lidar_basedon_pose, P0_camera, downsample_ratio, image=None):
        # in_image_feat: Tensor [B, C, H, W]
        # in_cloud_feat: ME.SparseTensor

        assert downsample_ratio % 4 == 0
        assert image.shape[-2] == in_image_feat.shape[-2] * downsample_ratio



        b,c,h,w = in_image_feat.shape
        assert c == self.in_image_feat_dim


        # boras: [16,19]convnext  [16,20]resnet
        in_image_feat = self.conv256_128(in_image_feat) 



            

        
        pts_in_image, pts_uv_in_image, pts_colors_in_image, pts_out_image, pts_feat_in_image, pts_feat_out_image = convert_pts_to_uv(
            in_cloud_feat, T_camera_lidar_basedon_pose, P0_camera)
        pts_uv_in_image[:,1:] = pts_uv_in_image[:,1:] * 0.125 / downsample_ratio



        # _batch_0_ids = pts_in_image[:,0]==0
        # _pts_in_image = pts_in_image[_batch_0_ids,1:]
        # _pts_out_image = pts_out_image[pts_out_image[:,0]==0,1:]
        # _pts = torch.cat([_pts_in_image, _pts_out_image], dim=0)
        # _colors = torch.cat([
        #     torch.tensor([0, 0, 0.2]).unsqueeze(0).repeat(_pts_in_image.shape[0],1),
        #     torch.tensor([0.7, 0, 0]).unsqueeze(0).repeat(_pts_out_image.shape[0],1)
        # ],dim=0)
        # viz_lidar_open3d(_pts.cpu().numpy(), _colors.cpu().numpy())
        # _uv = pts_uv_in_image[_batch_0_ids,1:]
        # _colors = pts_colors_in_image[_batch_0_ids]
        # # _image = in_image_feat[0]
        # _image = image[0]
        # viz_pts_projected_on_image(_uv*16, _colors, _image, dpi=300)



        pixel_uv, pixel_feat = convert_image_to_uv(in_image_feat)
        pixel_uv = pixel_uv.type_as(in_image_feat)
        pixel_feat = pixel_feat.type_as(in_image_feat)


        uv_all = torch.cat([pixel_uv, pts_uv_in_image], dim=0) # [b*h*w+N,3]
        feat_all = torch.cat([pixel_feat, pts_feat_in_image], dim=0) # [b*h*w+N,c]


        uv_all = torch.cat([
            uv_all, torch.ones([uv_all.shape[0], 1]).type_as(uv_all)
        ], dim=-1).int()

        
        assert uv_all[:,0].max() == b-1
        assert uv_all[:,1].max() < w+1
        assert uv_all[:,2].max() < h+1

        # feat_all: [N, c]
        # uv_all: [N, 3]int32 gradF
        input_tensor = sptr.SparseTrTensor(query_feats=feat_all, query_indices=uv_all, spatial_shape=None, batch_size=None)
        output_tensor = self.attn(input_tensor, use_proj=args.use_proj_inffblocal)


        if args.use_attninlocalffb:
            # if args.use_attn2:
            if args.gatt_fusion_block_type2 is not None:
                output_tensor2 = self.attn2(input_tensor)
                output_tensor2 = output_tensor2.query_feats
                output_tensor.query_feats += output_tensor2
            # if args.use_attn3:
            if args.gatt_fusion_block_type3 is not None:
                output_tensor3 = self.attn3(input_tensor)
                output_tensor3 = output_tensor3.query_feats
                output_tensor.query_feats += output_tensor3



        image_feat = output_tensor.query_feats[:len(pixel_uv)]
        pts_feat_in_image = output_tensor.query_feats[len(pixel_uv):]
        assert len(image_feat)+len(pts_feat_in_image) == len(feat_all)




        # feat_all_inout_image = torch.cat([image_feat, pts_feat_in_image, pts_feat_out_image], dim=0)



        if args.usemeconv1x1_aftersptr:
            if args.beforeafter_sharefc:
                None
                # feat_all_inout_image_F = feat_all_inout_image
                # feat_all_inout_image_identity  = feat_all_inout_image

                # if args.beforeafter_convtype == 'fc':
                #     feat_all_inout_image_F = self.outfc128_128(feat_all_inout_image_identity)
                # elif args.beforeafter_convtype == 'fcode':
                #     feat_all_inout_image_F = odeint(self.outfc128_128_ode, feat_all_inout_image_F, self.t, 
                #                                     method='dopri5', atol=self.tol, rtol=self.tol)[-1]
                #     if args.beforeafter_useres:
                #         feat_all_inout_image_F += feat_all_inout_image_identity
                #     if args.beforeafter_useextrafc:
                #         feat_all_inout_image_F += self.outfc128_128(feat_all_inout_image_identity)


                # image_feat = feat_all_inout_image_F[:len(pixel_uv)] 
                # pts_feat_in_image = feat_all_inout_image_F[len(pixel_uv):len(pixel_uv)+len(pts_feat_in_image)]
                # pts_feat_out_image = feat_all_inout_image_F[len(pixel_uv)+len(pts_feat_in_image):]
                # assert len(image_feat)+len(pts_feat_in_image) == len(feat_all)
                # assert len(pts_feat_in_image)+len(pts_feat_out_image) == len(in_cloud_feat.F)


            elif not args.beforeafter_sharefc:
                image_feat_identity = image_feat
                # pts_feat_inout_image_identity = torch.cat([pts_feat_in_image, pts_feat_out_image], dim=0)

                if args.beforeafter_convtype == 'fc':
                    image_feat_F = self.outimagefc128_128(image_feat_identity)
                    # pts_feat_inout_image_F = self.outfc128_128(pts_feat_inout_image_identity)
                elif args.beforeafter_convtype == 'fcode':
                    image_feat_F = odeint(self.outimagefc128_128_ode, image_feat_identity, self.t, 
                                                    method='dopri5', atol=self.tol, rtol=self.tol)[-1]
                    # pts_feat_inout_image_F = odeint(self.outfc128_128_ode, pts_feat_inout_image_identity, self.t, 
                    #                                 method='dopri5', atol=self.tol, rtol=self.tol)[-1]
                    
                    if args.beforeafter_useres:
                        image_feat_F += image_feat_identity
                        # pts_feat_inout_image_F += pts_feat_inout_image_identity
                    if args.beforeafter_useextrafc:
                        image_feat_F += self.outimagefc128_128(image_feat_identity)
                        # pts_feat_inout_image_F += self.outfc128_128(pts_feat_inout_image_identity)

                image_feat = image_feat_F
                # pts_feat_in_image = pts_feat_inout_image_F[:len(pts_feat_in_image)]
                # pts_feat_out_image = pts_feat_inout_image_F[len(pts_feat_in_image):]
                assert len(image_feat)+len(pts_feat_in_image) == len(feat_all)
                assert len(pts_feat_in_image)+len(pts_feat_out_image) == len(in_cloud_feat.F)




        if args.image_useswin:
            image_feat = image_feat.view(b,h,w,self.embed_dim) # [b,h,w,c]
            image_feat_identity = image_feat
            if args.imageswin_type == 'swin':
                image_feat_F = self.swinlayer(image_feat_identity)
            elif args.imageswin_type == 'swinode':
                image_feat_F = odeint(self.swinlayer_ode, image_feat_identity, self.t,
                                    method='dopri5', atol=self.tol, rtol=self.tol)[-1]
                if args.imageswin_useres:
                    image_feat_F += image_feat_identity
            elif 'beltrami' in args.imageswin_type:
                image_feat_identity = image_feat_identity.permute(0,3,1,2) # [b,c,h,w]
                image_feat_F = self.beltrami(image_feat_identity) # [b,c,h,w]
                image_feat_F = image_feat_F.permute(0,2,3,1) # [b,h,w,c]
                image_feat_identity = image_feat_identity.permute(0,2,3,1) # [b,h,w,c]

            image_feat = image_feat_F

                

        image_feat = image_feat.view(b,h*w,self.embed_dim).contiguous()
        image_feat = image_feat.permute(0,2,1)
        image_feat = image_feat.view(b,self.embed_dim,h,w).contiguous()
        




        output_image_feat = self.conv128_256(image_feat)








        cloud_feat = torch.cat([pts_feat_in_image, pts_feat_out_image],dim=0)
        cloud_coords = torch.cat([pts_in_image, pts_out_image],dim=0)
        assert len(cloud_feat) == len(cloud_coords)

        output_cloud_feat = ME.SparseTensor(cloud_feat, coordinates=cloud_coords,
                                            # tensor_stride=in_cloud_feat.tensor_stride
                                            )







        return output_image_feat, output_cloud_feat

        # output_dict = {
        #     'output_image_feat': output_image_feat,
        #     'output_cloud_feat': output_cloud_feat,
        # }
        # return output_dict
    


