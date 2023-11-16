# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.minkloc_multimodal import MinkLocMultimodal, ResnetFPN
from tools.utils import MinkLocParams
from models.minkloc_multimodal import GeneralFPN
from tools.utils import set_seed
set_seed(7)
from tools.options import Options
args = Options().parse()

def model_factory(params: MinkLocParams):
    in_channels = 1

    # MinkLocMultimodal is our baseline MinkLoc++ model producing 256 dimensional descriptor where
    # each modality produces 128 dimensional descriptor
    # MinkLocRGB and MinkLoc3D are single-modality versions producing 256 dimensional descriptor
    if params.model_params.model == 'MinkLocMultimodal':
        cloud_fe_size = args.minkloc_cloud_fe_dim
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64, 64], layers=[1, 1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')
        

        image_fe_size = args.minkloc_image_fe_dim

        if 'org' in args.minkloc_image_fe:
            image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                                fh_num_bottom_up=4, fh_num_top_down=0)
        else:
            image_fe = GeneralFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                                fh_num_bottom_up=4, fh_num_top_down=0)

        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size + image_fe_size)



    elif params.model_params.model == 'MinkLoc3D':
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')
        model = MinkLocMultimodal(cloud_fe, cloud_fe_size, None, 0, output_dim=cloud_fe_size,
                                  dropout_p=None)
        

    elif params.model_params.model == 'MinkLocRGB':
        image_fe_size = 256
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocMultimodal(None, 0, image_fe, image_fe_size, output_dim=image_fe_size)

        
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    return model
