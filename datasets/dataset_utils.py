# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset
from datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform
from datasets.samplers import BatchSampler
from tools.utils import MinkLocParams
try: from viz_lidar_mayavi_open3d import *
except: None

import matplotlib.pyplot as plt

import torchvision



from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)










def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)

    if params.use_rgb:
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()
    else:
        image_train_transform = None
        image_val_transform = None



    if args.dataset == 'oxford':
        datasets['train'] = OxfordDataset(args.dataset_folder, query_filename=params.train_file, image_path=args.image_path,
                                        lidar2image_ndx_path=params.lidar2image_ndx_path, transform=train_transform,
                                        set_transform=train_set_transform, image_transform=image_train_transform,
                                        use_cloud=params.use_cloud)
        val_transform = None
        if params.val_file is not None:
            datasets['val'] = OxfordDataset(args.dataset_folder, params.val_file, image_path=args.image_path,
                                            lidar2image_ndx_path=params.lidar2image_ndx_path, transform=val_transform,
                                            set_transform=train_set_transform, image_transform=image_val_transform,
                                            use_cloud=params.use_cloud)
            

    elif args.dataset == 'oxfordadafusion':
        datasets['train'] = OxfordDataset(args.dataset_folder, query_filename='oxfordadafusion_training_queries_baseline.pickle', image_path=args.image_path,
                                        lidar2image_ndx_path=params.lidar2image_ndx_path, transform=train_transform,
                                        set_transform=train_set_transform, image_transform=image_train_transform,
                                        use_cloud=params.use_cloud)
        val_transform = None
        if params.val_file is not None:
            datasets['val'] = OxfordDataset(args.dataset_folder, params.val_file, image_path=args.image_path,
                                            lidar2image_ndx_path=params.lidar2image_ndx_path, transform=val_transform,
                                            set_transform=train_set_transform, image_transform=image_val_transform,
                                            use_cloud=params.use_cloud)
    
    

    elif args.dataset == 'boreas':
        datasets['train'] = OxfordDataset(args.dataset_folder, query_filename='boreas_training_queries_baseline.pickle', image_path=args.image_path,
                                lidar2image_ndx_path=params.lidar2image_ndx_path, transform=train_transform,
                                set_transform=train_set_transform, image_transform=image_train_transform,
                                use_cloud=params.use_cloud)
        # val_transform = None
        # if params.val_file is not None:
        #     datasets['val'] = OxfordDataset(args.dataset_folder, query_filename='boreas_lidar2image_ndx.pickle', image_path=args.image_path,
        #                                     lidar2image_ndx_path=params.lidar2image_ndx_path, transform=val_transform,
        #                                     set_transform=train_set_transform, image_transform=image_val_transform,
        #                                     use_cloud=params.use_cloud)



    # a = len(datasets['train'])
    # b = len(datasets['val'])

    return datasets


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'clouds' in data_list[0]:

            coords = [e['coords'] for e in data_list]
            clouds = [e['clouds'] for e in data_list]



            # # normalized version
            # if args.dataset in ['oxford','oxfordadafusion']:
            #     coords_and_feat_clouds = [ME.utils.sparse_quantize(coordinates=e, features=e, quantization_size=mink_quantization_size)
            #             for e in _clouds_normalized_list]
            # elif args.dataset == 'boreas':
            #     coords_and_feat_clouds = [ME.utils.sparse_quantize(coordinates=e, features=e, quantization_size=1)
            #             for e in _clouds_normalized_list]


            coords = ME.utils.batched_coordinates(coords)
            clouds = torch.cat(clouds, dim=0)
            assert coords.shape[0]==clouds.shape[0]
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)



            result['coords'] = coords
            result['clouds'] = clouds
            result['features'] = feats


        if 'uv' in data_list[0]:
            uv = [e['uv'] for e in data_list]
            result['uv'] = torch.cat(uv, dim=0)
            assert result['uv'].shape[0] == coords.shape[0]

        if 'colors' in data_list[0]:
            colors = [e['colors'] for e in data_list]
            result['colors'] = torch.cat(colors, dim=0)
            assert result['colors'].shape[0] == coords.shape[0]

        if 'maskselect' in data_list[0]:
            maskselect = [e['maskselect'] for e in data_list]
            result['maskselect'] = torch.cat(maskselect, dim=0)
            assert result['maskselect'].shape[0] == coords.shape[0]


        if 'P0_camera' in data_list[0]:
            P0_camera = [e['P0_camera'] for e in data_list]
            result['P0_camera'] = torch.stack(P0_camera, dim=0)


        if 'T_camera_lidar_basedon_pose' in data_list[0]:
            T_camera_lidar_basedon_pose = [e['T_camera_lidar_basedon_pose'] for e in data_list]
            result['T_camera_lidar_basedon_pose'] = torch.stack(T_camera_lidar_basedon_pose, dim=0)


        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0)       # Produces (N, C, H, W) tensor






        




        if 'sph_cloud' in data_list[0]:
            sph_clouds = [e['sph_cloud'] for e in data_list]
            result['sph_cloud'] = torch.stack(sph_clouds, dim=0)       # Produces (N, C, H, W) tensor

        

        return result

    return collate_fn








def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], 
                                #  batch_size=params.batch_size,
                                #  batch_size_limit=params.batch_size_limit,
                                 batch_size = args.train_batch_size,
                                 batch_size_limit = args.train_batch_size,
                                 batch_expansion_rate = params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  params.model_params.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=args.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], 
                                #    batch_size=params.val_batch_size
                                   batch_size=args.val_batch_size
                                   )
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], params.model_params.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=args.num_workers, pin_memory=True)

    return dataloders


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
