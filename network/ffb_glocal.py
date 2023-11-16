




import torch
import torch.nn as nn
from third_party.SparseTransformer import sptr
import MinkowskiEngine as ME

from layers.pooling import MinkSpconvGeM
from network.resnetfpn_simple import ImageGeM
import torch.nn.functional as F


class FFBGLocal(nn.Module):
    def __init__(self, embed_dim, image_dim,):
        super().__init__()


        self.attn = sptr.VarLengthMultiheadSA(
                                        embed_dim=embed_dim, 
                                        num_heads=8, 
                                        indice_key='None', 
                                        window_size=1
                                        )
        
        self.fc_256_128 = nn.Linear(image_dim, embed_dim)
        self.fc_128_256 = nn.Linear(embed_dim, image_dim)

        self.image_gem = ImageGeM()
        self.cloud_gem = MinkSpconvGeM(input_dim=embed_dim)


    def forward(self, image_feat, cloud_feat):
        image_feat_org = image_feat
        cloud_feat_org = cloud_feat

        assert isinstance(image_feat, torch.Tensor)
        assert isinstance(cloud_feat, ME.SparseTensor)

        assert len(image_feat.shape) == 4
        image_feat = image_feat.permute(0,2,3,1) # (B, H, W, C)
        image_feat = self.fc_256_128(image_feat) 
        # b, c, h, w = image_feat.shape

        image_ids = torch.arange(image_feat.shape[0]).to(image_feat.device) # (B)
        image_ids = image_ids.unsqueeze(-1) # (B, 1)
        image_ids = image_ids.repeat(1, image_feat.shape[1]*image_feat.shape[2]) # (B, H*W)
        image_ids = image_ids.view(-1, 1) # (B*H*W, 1)
        image_uvs = torch.zeros_like(image_ids).repeat(1, 3) # (B*H*W, 3)
        image_ids = torch.cat([image_ids, image_uvs], dim=1) # (B*H*W, 4)
        image_feat = image_feat.view(-1, image_feat.shape[-1]) # (B*H*W, C)

        cloud_ids = cloud_feat.C
        cloud_feat = cloud_feat.F

        fusion_feat = torch.cat([image_feat, cloud_feat], dim=0)
        fusion_feat = F.normalize(fusion_feat, dim=-1, p=2)
        fusion_ids = torch.cat([image_ids, cloud_ids], dim=0)

        # feat_all: [N, c]
        # uv_all: [N, 3]int32 gradF
        input_tensor = sptr.SparseTrTensor(query_feats=fusion_feat, query_indices=fusion_ids, spatial_shape=None, batch_size=None)
        output_tensor = self.attn(input_tensor, use_proj=False)

        output_image_feat = output_tensor.query_feats[:image_feat.shape[0]]
        output_image_feat = output_image_feat.view(image_feat_org.shape[0], image_feat_org.shape[2], image_feat_org.shape[3], -1) # (B, H, W, C)
        output_image_feat = self.fc_128_256(output_image_feat)
        output_image_feat = output_image_feat.permute(0,3,1,2) # (B, C, H, W)
        assert output_image_feat.shape == image_feat_org.shape

        output_cloud_feat = output_tensor.query_feats[image_feat.shape[0]:]
        output_cloud_feat = ME.SparseTensor(output_cloud_feat, coordinates=cloud_feat_org.C)


        output_image_feat_gem = self.image_gem(output_image_feat)
        output_cloud_feat_gem = self.cloud_gem(output_cloud_feat)

        output_dict = {
            'image_feat': output_image_feat,
            'cloud_feat': output_cloud_feat,
            'image_feat_gem': output_image_feat_gem,
            'cloud_feat_gem': output_cloud_feat_gem,
        }

        return output_dict