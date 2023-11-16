# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from tools.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform

from datasets.dataloader_dataset import *

from network.univpr_v2 import UniVPRV2

DEBUG = False








def evaluate(model, device, params, silent=True):
    # Run evaluation on all eval datasets
    assert len(params.eval_database_files) == len(params.eval_query_files)


    lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))

    stats = {}
    # only {'oxford_evaluation_database.pickle', ['oxford_evaluation_query.pickle']}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        # location_name = args.dataset 
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        if args.dataset in ['oxford','oxfordadafusion']:
            p = os.path.join(params.dataset_folder, query_file)
            with open(p, 'rb') as f:
                query_sets = pickle.load(f)
        elif args.dataset == 'boreas':
            # query_sets = database_sets.copy()
            p = os.path.join(params.dataset_folder, query_file)
            with open(p, 'rb') as f:
                query_sets = pickle.load(f)


        temp = evaluate_dataset(model, device, params, database_sets, query_sets, silent=silent, lidar2image_ndx=lidar2image_ndx)
        stats[location_name] = temp

    return stats









def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True, lidar2image_ndx=None):


    # Run evaluation on a single dataset

    if args.dataset == 'oxfordadafusion':
        recall = np.zeros(20)
    else:
        recall = np.zeros(25)
        
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()






    # -- new
    if args.dataset == 'boreas':
        database_embeddings = get_latent_vectors_with_merged(model, database_sets, device, params, lidar2image_ndx)
        query_embeddings = database_embeddings.copy()
        print('boreas direct copy')
    else:
        database_embeddings = get_latent_vectors_with_merged(model, database_sets, device, params, lidar2image_ndx)
        query_embeddings = get_latent_vectors_with_merged(model, query_sets, device, params, lidar2image_ndx)





    # for i in range(len(query_sets)):

    recall_for_each_scene = np.zeros([len(query_sets), 25])

    for i in tqdm.tqdm(range(len(query_sets)), disable=silent):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

            recall_for_each_scene[i] += np.array(pair_recall)


    # count 23*22=506
    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)

    ave_recall_for_each_scene = recall_for_each_scene / (len(query_sets) - 1)






    stats = {'ave_one_percent_recall': ave_one_percent_recall, 
             'ave_recall': ave_recall,
             'average_similarity': average_similarity,
             'ave_recall_for_each_scene':ave_recall_for_each_scene,
             }
    


    return stats










def get_latent_vectors_with_merged(model, sets_unmerged, device, params, lidar2image_ndx):
        # Adapted from original PointNetVLAD code
    sets_merged = []
    scene_sets_lengths = []
    for scene_set in sets_unmerged:
        scene_sets_lengths.append(len(scene_set))
        for eachitem in scene_set.values():
            sets_merged.append(eachitem)



    if DEBUG:
        embeddings = np.random.rand(len(sets_merged), 256)
        return embeddings


    dataset = DataloaderDataset(sets_merged, device, params, lidar2image_ndx)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=args.val_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn,
                                 )


    model.eval()
    embeddings_l = []


    for i_batch, batch_dict in tqdm.tqdm(enumerate(dataloader)):


        batch_dict = {e: batch_dict[e].to(device) for e in batch_dict}


        with torch.no_grad():
            output = model(batch_dict)

        embedding = output['embedding']

        # embedding is (1, 256) tensor
        if params.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    assert len(embeddings)==len(dataset)



    current_id = 0
    embeddings_unmerged = []
    for length in scene_sets_lengths:
        embeddings_this_scene = embeddings[current_id:current_id+length]
        embeddings_unmerged.append(embeddings_this_scene)
        current_id += length


    assert len(sets_merged) == sum(scene_sets_lengths)
    assert len(embeddings_unmerged) == len(sets_unmerged)
    


    return embeddings_unmerged








def get_latent_vectors(model, set, device, params, lidar2image_ndx):
        # Adapted from original PointNetVLAD code

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings


    dataset = DataloaderDataset(set, device, params, lidar2image_ndx)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=params.val_batch_size,
                                 shuffle=False,
                                 num_workers=params.num_workers,
                                 collate_fn=collate_fn
                                 )


    model.eval()
    embeddings_l = []


    for i_batch, batch_dict in enumerate(dataloader):
        None


        batch_dict['coords'] = batch_dict['coords'] .to(device)
        batch_dict['features'] = batch_dict['features'] .to(device)
        batch_dict['images'] = batch_dict['images'] .to(device)
        batch_dict['clouds'] = batch_dict['clouds'] .to(device)



        with torch.no_grad():
            output = model(batch_dict)

        embedding = output['embedding']

        # embedding is (1, 256) tensor
        if params.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    assert len(embeddings)==len(dataset)



    return embeddings








def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    if args.dataset == 'oxfordadafusion':
        num_neighbors = 20
    else:
        num_neighbors = 25

    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)



    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)




        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])








if __name__ == "__main__":




    from tools.options import Options
    args = Options().parse()



    epoch = 0
    for epoch in [0]:

        
        # args.weights = os.path.join(args.models_dir, f'ep{epoch}_{iter}.pth')
        # args.weights = os.path.join(args.models_dir, f'ep{epoch}.pth')

        # args is set as the dataset to evaluate on
        # replace it with the dataset the weights belong to
        # args.weights = args.weights.replace('daoxianglake', 'boreas')
        args.weights = args.weights.replace('boreas', 'boreas')



        print('Config path: {}'.format(args.config))
        print('Model config path: {}'.format(args.model_config))
        if args.weights is None:
            w = 'RANDOM WEIGHTS'
        else:
            w = args.weights
        print('Weights: {}'.format(w))
        print('')



        params = MinkLocParams(args.config, args.model_config)
        params.print()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print('Device: {}'.format(device))



        # model = model_factory(params)

        if args.use_minkloc:
            model = model_factory(params)
        else:
            model = UniVPRV2()




        if args.weights is not None:
            assert os.path.exists(args.weights), 'Weights do not exist: {}'.format(args.weights)
            print('Loading weights: {}'.format(args.weights))
            model.load_state_dict(torch.load(args.weights, map_location=device))


        model.to(device)

        test_stats = evaluate(model, device, params, silent=False)
        # print_eval_stats(stats)


        test_stats = test_stats[args.dataset]
        recall_top1p = test_stats['ave_one_percent_recall']
        recall_topN = test_stats['ave_recall']
        recall_top1 = recall_topN[0]

        ave_recall_for_each_scene = test_stats['ave_recall_for_each_scene']
        ave_recall_for_each_scene_top1 = ave_recall_for_each_scene[:,0]

        
        print(f'recall_top1p                       {recall_top1p:.2f}')
        print(f'recall_top1                        {recall_top1:.2f}')
        print(f'ave_recall_for_each_scene_top1     {ave_recall_for_each_scene_top1.round(2)}')