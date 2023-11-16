
from tools.utils import set_seed
set_seed(7)

import os
import torch
import tqdm


from evaluate import evaluate
from tools.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory
from models.minkloc_multimodal import MinkLocMultimodal
import time
from network.univpr_v2 import UniVPRV2
from tqdm import tqdm

import os


from tools.options import Options
args = Options().parse()
from tools.utils import set_seed
set_seed(7)

VERBOSE = False



import torch
from tools.utils import *
from tools.options import Options
from datasets.dataset_utils import make_dataloaders
from tools.utils import set_seed
set_seed(7)
args = Options().parse()










def test_after_epoch(model, device, params, stats):
    # Evaluate the final model
    model.eval()
    test_stats = evaluate(model, device, params, silent=False)
    # test_stats = test_stats['oxford']
    test_stats = test_stats[args.dataset]
    recall_top1p = test_stats['ave_one_percent_recall']
    recall_topN = test_stats['ave_recall']
    recall_top1 = recall_topN[0]


    ave_recall_for_each_scene = test_stats['ave_recall_for_each_scene']
    ave_recall_for_each_scene_top1 = ave_recall_for_each_scene[:,0]

    
    # print(f'recall_top1p                       {recall_top1p:.2f}')
    # print(f'recall_top1                        {recall_top1:.2f}')
    # print(f'ave_recall_for_each_scene_top1     {ave_recall_for_each_scene_top1.round(2)}')


    return recall_top1p, recall_top1, ave_recall_for_each_scene_top1



def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Loss (mean/total): {:.4f} / {:.4f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['total_loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

    if 'final_loss' in stats:
        # Multi loss
        s1 = '{} - Loss (total/final'.format(phase)
        s2 = '{:.4f} / {:.4f}'.format(stats['loss'], stats['final_loss'])
        s3 = 'Active triplets (final '
        s4 = '{:.1f}'.format(stats['final_num_non_zero_triplets'])
        if 'cloud_loss' in stats:
            s1 += '/cloud'
            s2 += '/ {:.4f}'.format(stats['cloud_loss'])
            s3 += '/cloud'
            s4 += '/ {:.1f}'.format(stats['cloud_num_non_zero_triplets'],)
        if 'image_loss' in stats:
            s1 += '/image'
            s2 += '/ {:.4f}'.format(stats['image_loss'])
            s3 += '/image'
            s4 += '/ {:.1f}'.format(stats['image_num_non_zero_triplets'],)

        s1 += '): '
        s3 += '): '
        print(s1 + s2)
        print(s3 + s4)


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(dataloaders, params: MinkLocParams, debug=False):
    # Create model class

    if args.use_minkloc:
        model = model_factory(params)
    else:
        model = UniVPRV2()


    num_parameters = sum([e.nelement() for e in model.parameters()])
    with open(f'results/{args.exp_name}.txt', 'a') as f:
        f.write(f'Number of parameters: {num_parameters}\n')
        print(f'Number of parameters: {num_parameters}')




    model_name = args.exp_name


    print('Model name: {}'.format(model_name))



    params_l = []
    if isinstance(model, MinkLocMultimodal):
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_fe is not None:
            params_l.append({'params': model.image_fe.parameters(), 'lr': args.image_lr})
        if model.cloud_fe is not None:
            params_l.append({'params': model.cloud_fe.parameters(), 'lr': args.cloud_lr})
        if model.final_block is not None:
            params_l.append({'params': model.final_net.parameters(), 'lr': args.cloud_lr})
    else:
        # All parameters use the same lr
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_fe is not None:
            params_l.append({'params': model.image_fe.parameters(), 'lr': args.image_lr}) # 1e-4
        if model.cloud_fe is not None:
            params_l.append({'params': model.cloud_fe.parameters(), 'lr': args.cloud_lr}) # 1e-3
        if model.ffbs is not None:
            params_l.append({'params': model.ffbs.parameters(), 'lr': args.image_lr})
        if model.laterimagebranches is not None:
            params_l.append({'params': model.laterimagebranches.parameters(), 'lr': args.image_lr})
        if model.latercloudbranches is not None:
            params_l.append({'params': model.latercloudbranches.parameters(), 'lr': args.cloud_lr})
        if hasattr(model, 'sph_cloud_fe'):
            params_l.append({'params': model.sph_cloud_fe.parameters(), 'lr': args.cloud_lr})
        if model.ffblocals is not None:
            params_l.append({'params': model.ffblocals.parameters(), 'lr': args.image_lr})


    



    # -- loss function
    loss_fn = make_loss(params)


    # Training elements
    if params.optimizer == 'Adam':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.Adam(params_l)
        else:
            optimizer = torch.optim.Adam(params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'SGD':
        # SGD with momentum (default momentum = 0.9)
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.SGD(params_l)
        else:
            optimizer = torch.optim.SGD(params_l, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(params.optimizer))
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))




    # -- device
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"
    print('Model device: {}'.format(device))



    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}



    for epoch in range(args.epochs):
        t0 = time.time()
        txt = []

        model.train()



        for i_batch, batch_dict in tqdm(enumerate(dataloaders['train'])):


            batch_dict = {e: batch_dict[e].to(device) for e in batch_dict}

            positives_mask = batch_dict['positives_mask']
            negatives_mask = batch_dict['negatives_mask']
            n_positives = torch.sum(positives_mask).item()
            n_negatives = torch.sum(negatives_mask).item()
            if n_positives == 0 or n_negatives == 0:
                # Skip a batch without positives or negatives
                print('WARNING: Skipping batch without positive or negative examples')
                continue

            optimizer.zero_grad()


            embeddings = model(batch_dict)
            loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)


            loss.backward()
            optimizer.step()




            if isinstance(model, MinkLocMultimodal):
                now_image_lr = optimizer.param_groups[0]["lr"]
                now_cloud_lr = optimizer.param_groups[1]["lr"]
                now_fusion_lr = 0
            else:
                now_image_lr = optimizer.param_groups[0]["lr"]
                now_cloud_lr = optimizer.param_groups[1]["lr"]
                now_fusion_lr = optimizer.param_groups[2]["lr"]


            torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors





        if scheduler is not None:
            scheduler.step()




        model.eval()
        recall_top1p, recall_top1, ave_recall_for_each_scene_top1 = test_after_epoch(model, device, params, stats)




        txt.append(f'recall_top1p                     {recall_top1p:.2f}')
        txt.append(f'recall_top1                      {recall_top1:.2f}')
        txt.append(f'ave_recall_for_each_scene_top1   {ave_recall_for_each_scene_top1.round(2)}')
        txt.append(f'Epoch {epoch}\t  Lr {now_image_lr}_{now_cloud_lr}_{now_fusion_lr}\t  Time {(time.time()-t0):.2f}')
        txt.append(f'-------------------------------------------------{args.exp_name}') 




        with open('results.txt', 'a') as f:
            for each_line in txt:
                f.writelines(each_line)
                f.writelines('\n')
                print(each_line)

        with open(f'results/{args.exp_name}.txt', 'a') as f:
            for each_line in txt:
                f.writelines(each_line)
                f.writelines('\n')








if __name__ == '__main__':

    datetime_start = get_datetime()
    if not os.path.exists('results'):
        os.mkdir('results')

    with open('results.txt', 'w') as f:
        f.writelines('\n')

    with open(f'results/{args.exp_name}.txt', 'w') as f:
        f.writelines('\n')




    params = MinkLocParams(args.config, args.model_config)

    
    # params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)




    dataloaders = make_dataloaders(params, debug=args.debug)
    


    do_train(dataloaders, params, debug=args.debug)



    datetime_end = get_datetime()
    with open('results.txt', 'a') as f:
        f.writelines('\n')
        f.writelines(datetime_start)
        f.writelines('\n')
        f.writelines(datetime_end)

    with open(f'results/{args.exp_name}.txt', 'a') as f:
        f.writelines('\n')
        f.writelines(datetime_start)
        f.writelines('\n')
        f.writelines(datetime_end)






