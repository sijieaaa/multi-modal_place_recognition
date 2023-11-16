


import torch.utils.data as data
import MinkowskiEngine as ME
import torch
import os

# from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform
import numpy as np
import pickle
import random
import torchvision.transforms.functional as TVF
from PIL import Image
from datasets.oxford import ts_from_filename
from tools.utils import *
from tools.options import *
from viz_lidar_mayavi_open3d import *
args = Options().parse()
set_seed(7)


def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    lidar_ts, traversal = ts_from_filename(filename)
    assert lidar_ts in lidar2image_ndx, 'Unknown lidar timestamp: {}'.format(lidar_ts)

    # Randomly select one of images linked with the point cloud
    if k is None or k > len(lidar2image_ndx[lidar_ts]):
        k = len(lidar2image_ndx[lidar_ts])

    image_ts = random.choice(lidar2image_ndx[lidar_ts][:k])
    image_file_path = os.path.join(image_path, traversal, str(image_ts) + image_ext)
    #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
    img = Image.open(image_file_path)
    return img




def load_data_item(image_path, cloud_path):
    # returns Nx3 matrix

    # image_path 
    # cloud_path = os.path.join(params.dataset_folder, cloud_name)


    result_dict = {}
    if True:
        pc = np.fromfile(cloud_path, dtype=np.float32)
        # coords are within -1..1 range in each dimension
        # assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(cloud_path)
        # pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = np.reshape(pc, [-1,4])
        # viz_lidar_open3d(pc[:,:3])
        pc = torch.tensor(pc, dtype=torch.float)
        result_dict['coords'] = pc
        result_dict['clouds'] = pc.detach().clone()

    if True:
        # Get the first closest image for each LiDAR scan
        # assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
        # img = image4lidar(cloud_name, params.image_path, '.png', lidar2image_ndx, k=1)


        img = Image.open(image_path)
        # img = TVF.resize(img,size=200)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result_dict['image'] = transform(img)

    return result_dict





def collate_fn_kitti(batch_list):

    batch_dict = {}
    coords_list = []
    images_list = []
    clouds_list = []

    for each_batch in batch_list:
        coords = each_batch['coords'] # [4096,3]
        images = each_batch['images']
        # clouds = each_batch['clouds']

        coords = ME.utils.sparse_quantize(coordinates=coords, quantization_size=0.25)

        coords_list.append(coords)
        images_list.append(images)
        # clouds_list.append(clouds)


    coords_list = ME.utils.batched_coordinates(coords_list)
    features_list = torch.ones([len(coords_list), 1])
    images_list = torch.stack(images_list)
    # clouds_list = torch.stack(clouds_list)


    batch_dict = {
        'coords': coords_list,
        'features': features_list,
        'images': images_list,
        # 'clouds': clouds_list
    }

    return batch_dict






class DataloaderDatasetKITTI(data.Dataset):
    def __init__(self, image_paths, cloud_paths):



        self.image_paths = image_paths

        self.cloud_paths = cloud_paths
        assert len(self.image_paths) == len(self.cloud_paths)




    def __len__(self):
        length = len(self.image_paths)

        return length




    def __getitem__(self, index):

        image_path = self.image_paths[index]
        cloud_path = self.cloud_paths[index]

        
        result_dict = load_data_item(image_path, cloud_path)


        data_dict = {}
        # -- no quantize
        data_dict['coords'] = result_dict['coords'][:,:3]




        data_dict['images'] = result_dict['image']




        data_dict['clouds'] = result_dict['clouds']



        return data_dict


