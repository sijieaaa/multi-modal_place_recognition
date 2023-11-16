import argparse
import os



class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):



        self.parser.add_argument('--cuda', type=str, default='0')
        self.parser.add_argument('--tryid', type=int, default=0)
        self.parser.add_argument('--save_weights', type=str, default=False)
        self.parser.add_argument('--num_workers', type=int, default=8)
        # oxford   boreas
        self.parser.add_argument('--dataset', type=str, 
                                 default='boreas'
                                 )

        self.parser.add_argument('--dataset_folder', type=str, 
                                 default='/home/sijie/vpr/BenchmarkBoreasv3',
                                 )
        self.parser.add_argument('--image_path', type=str, 
                                 default='/home/sijie/vpr/BenchmarkBoreasv3/boreas',
                                 )
        self.parser.add_argument('--n_points_boreas', type=int, default=4096) # only for boreas



        # False True
        self.parser.add_argument('--use_minkloc', type=str, default=False)
        self.parser.add_argument('--minkloc_image_fe_dim', type=int, default=256)
        self.parser.add_argument('--minkloc_cloud_fe_dim', type=int, default=128)

        
        self.parser.add_argument('--minkloc_image_fe', type=str, default='resnet18')
        # minkfpn  generalminkfpn
        self.parser.add_argument('--minkfpn', type=str, default='minkfpn')



        self.parser.add_argument('--use_ffblocal', type=str, default=True)

        
        self.parser.add_argument('--ffblocal_windowsize', type=int, default=1)
        self.parser.add_argument('--use_proj_inffblocal', type=str, default=False)
        self.parser.add_argument('--usemeconv1x1_beforesptr', type=str, default=False)
        self.parser.add_argument('--usemeconv1x1_aftersptr', type=str, default=False)
        # conv  fc  fcode  fcode64
        self.parser.add_argument('--beforeafter_convtype', type=str, default='fcode')
        self.parser.add_argument('--beforeafter_useres', type=str, default=False)     # only for fcode
        self.parser.add_argument('--beforeafter_useextrafc', type=str, default=False) # only for fcode
        self.parser.add_argument('--beforeafter_sharefc', type=str, default=False) # only for fcode

        self.parser.add_argument('--image_useswin', type=str, default=True) 
        self.parser.add_argument('--imageswin_windowsize', type=int, default=4) 
        self.parser.add_argument('--imageswin_useproj', type=str, default=False) 
        self.parser.add_argument('--imageswin_userelu', type=str, default=False) 
        # swin   swinode  beltrami_k?  beltramiv2_k?   beltramiII_k?   beltramiI_k?
        self.parser.add_argument('--imageswin_type', type=str, default='beltramiI_k25') 
        self.parser.add_argument('--imageswin_useres', type=str, default=False) 




        self.parser.add_argument('--image_fe', type=str, default='convnext_tiny')
        self.parser.add_argument('--image_fe_dim', type=int, default=128)
        # convnext_tiny[3,3,9,3]  
        self.parser.add_argument('--num_other_stage_blocks', type=int, default=3)
        self.parser.add_argument('--num_stage3_blocks', type=int, default=2)
        self.parser.add_argument('--sph_cloud_fe', type=str, default=None)
        # general_minkfpn   minkloc  
        self.parser.add_argument('--cloud_fe', type=str, default='minkloc')
        self.parser.add_argument('--cloud_fe_dim', type=int, default=128)
        # None   rand   globalavg   globalmax   globalavgmax
        self.parser.add_argument('--sample_mode', type=str, default='rand')
        self.parser.add_argument('--num_image_points', type=int, default=16) # convnext: oxford[15*20]=300   boreas[16*19]=304
        self.parser.add_argument('--num_cloud_points', type=int, default=16)
        self.parser.add_argument('--fusion_dim', type=int, default=128)
        # ln
        self.parser.add_argument('--gattnorm', type=str, default='ln')
        # gelu  relu
        self.parser.add_argument('--gattactivation', type=str, default='relu')
        # 4096  6144  8192   



        self.parser.add_argument('--num_ffbs', type=int, default=2)
        self.parser.add_argument('--num_blocks_in_later_image', type=int, default=1)
        self.parser.add_argument('--num_blocks_in_later_cloud', type=int, default=1)
        self.parser.add_argument('--num_blocks_in_ffb', type=int, default=1) # exclude the first block
        self.parser.add_argument('--use_l2norm_before_fusion', type=str, default=True)
        self.parser.add_argument('--use_a_fusion_first', type=bool, default=True)
        # gemextra   randomsamelength   gemextra_randomsamelength
        self.parser.add_argument('--in_cloud_feat_type', type=str, default='randomsamelength')
        # pure   mix   imageonly
        self.parser.add_argument('--ffb_type', type=str, default='pure')
        # basicblock  bottleneck  gatt  gattm   attn  resattn   swinblock
        self.parser.add_argument('--gatt_image_block_type', type=str, default='swinblock')
        # gatt  gattm   attn  resattn
        self.parser.add_argument('--gatt_cloud_block_type', type=str, default='attn')


        self.parser.add_argument('--use_attninlocalffb', type=str, default=False)
        self.parser.add_argument('--qkvmlp_layers', type=int, default=2)
        # euler  rk4  dopri5  midpoint  id
        self.parser.add_argument('--odeint_method', type=str, default='dopri5')
        self.parser.add_argument('--tol', type=float, default=1e-3)
        # id  relu  sigmoid  tanh
        self.parser.add_argument('--gatt_fusion_act_type', type=str, default='relu')
        self.parser.add_argument('--num_orders', type=int, default=1)
        self.parser.add_argument('--hyp_c', type=float, default=0.1)
        self.parser.add_argument('--hyp_c_train', type=str, default=False)
        self.parser.add_argument('--hyp_mode', type=str, default='e')
        self.parser.add_argument('--gatt_fusion_block_type', type=str, default='ooqkvg1')
        self.parser.add_argument('--gatt_fusion_block_type2', type=str, default=None)
        self.parser.add_argument('--gatt_fusion_block_type3', type=str, default=None)
        self.parser.add_argument('--ffblocal_block_type', type=str, default='ooqkvg1')


        # basicblock  bottleneck  gatt  gattm   attn  resattn   swinblock
        self.parser.add_argument('--later_image_branch_type', type=str, default='basicblock')
        # basicblock  submbasicblock
        self.parser.add_argument('--later_cloud_branch_type', type=str, default='basicblock')
        # add     add_w   times_sigmoid  times_wsigmoid   None
        self.parser.add_argument('--later_image_branch_interaction_type', type=str, default='add')
        self.parser.add_argument('--later_cloud_branch_interaction_type', type=str, default='add')
        # fconfusion   fconfusiongem
        self.parser.add_argument('--make_image_fusion_same_channel', type=str, default='fconfusiongem')
        # imageorg_cloudorg_ffb   imagelater_cloudlater_ffb   imagelater_cloudlater
        # imageorg    cloudorg
        # imagelater_cloudlater_ffb_sphcloud
        self.parser.add_argument('--final_embedding_type', type=str, default='imagelater_cloudlater_ffb')
        # image   cloud   cat
        self.parser.add_argument('--useminkloc_final_embedding_type', type=str, default='cat')






        self.parser.add_argument('--epochs', type=int, default=80)
        self.parser.add_argument('--train_batch_size', type=int, default=80) # boreas-80
        self.parser.add_argument('--val_batch_size', type=int, default=160)
        self.parser.add_argument('--image_lr', type=float, default=1e-4)
        self.parser.add_argument('--cloud_lr', type=float, default=1e-3)
        self.parser.add_argument('--fusion_lr', type=float, default=1e-4)




        # -- image augmentation rate
        self.parser.add_argument('--bcs_aug_rate', type=float, default=0.2) # 0.2
        self.parser.add_argument('--hue_aug_rate', type=float, default=0.1) # 0.1



        self.parser.add_argument('--config', type=str, default='config/config_baseline_multimodal.txt')
        self.parser.add_argument('--model_config', type=str, default='models/minklocmultimodal.txt')
        self.parser.add_argument('--mink_quantization_size', type=float, default=0.01)






        self.parser.add_argument('--resume_epoch', type=int, default=-1)




        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')



        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')


        self.parser.add_argument('--debug', type=str, default=False)

        





    def parse(self):
        self.initialize()
        self.args = self.parser.parse_args()

        args_dict = vars(self.args)
        # print(args_dict)
        for k, v in args_dict.items():
            if v=='False':
                args_dict[k] = False
            elif v=='True':
                args_dict[k] = True
            elif v=='None':
                args_dict[k] = None

        self.args = argparse.Namespace(**args_dict)


        if self.args.dataset == 'oxford':
            self.args.use_ffblocal = False


        
        self.args.exp_name = ''
        self.args.exp_name += f'{self.args.tryid}'

        # -- tune swin
        if self.args.use_minkloc:
            self.args.exp_name += f'__{self.args.dataset}'
            self.args.exp_name += f'__use_minkloc'
            self.args.exp_name += f'__{self.args.epochs}'
            self.args.exp_name += f'__{self.args.train_batch_size}'
            self.args.exp_name += f'__{self.args.useminkloc_final_embedding_type}'
            self.args.exp_name += f'__{self.args.minkloc_image_fe}'
            self.args.exp_name += f'__{self.args.minkloc_image_fe_dim}'
            self.args.exp_name += f'__{self.args.minkloc_cloud_fe_dim}'
        else:
            self.args.exp_name += f'__{self.args.dataset}'
            self.args.exp_name += f'__{self.args.image_fe}'
            self.args.exp_name += f'__{self.args.epochs}'
            self.args.exp_name += f'__{self.args.train_batch_size}'
            self.args.exp_name += f'__{self.args.num_other_stage_blocks}'
            self.args.exp_name += f'__{self.args.num_stage3_blocks}'
            self.args.exp_name += f'__{self.args.final_embedding_type}'
            self.args.exp_name += f'__ffblocal{self.args.use_ffblocal}'
            self.args.exp_name += f'__nffb{self.args.num_ffbs}'


            self.args.exp_name += f'__f{self.args.gatt_fusion_block_type}'
            if self.args.gatt_fusion_block_type == 'qkvmlp':
                self.args.exp_name += f'l{self.args.qkvmlp_layers}'


            

            self.args.exp_name += f'__{self.args.sample_mode}'
            if self.args.sample_mode == 'rand':
                self.args.exp_name += f'{self.args.num_image_points}'
                self.args.exp_name += f'_{self.args.num_cloud_points}'

            if 'ode' in self.args.gatt_fusion_block_type:
                self.args.exp_name += f'__{self.args.odeint_method}'



            if 'hypqkv' in self.args.gatt_fusion_block_type:
                self.args.exp_name += f'__c{self.args.hyp_c}'
                self.args.exp_name += f'{self.args.hyp_c_train}'
                self.args.exp_name += f'_{self.args.hyp_mode}'

            elif 'ooqkv' in self.args.gatt_fusion_block_type:
                self.args.exp_name += f'_{self.args.gatt_fusion_act_type}'
                self.args.exp_name += f'_int{self.args.odeint_method}'


            if self.args.dataset in ['boreas']:
                if self.args.use_ffblocal:
                    self.args.exp_name += f'__ffblocal{self.args.ffblocal_block_type}'
                    if 'ooqkv' in self.args.ffblocal_block_type:
                        self.args.exp_name += f'_{self.args.gatt_fusion_act_type}'
                        self.args.exp_name += f'_int{self.args.odeint_method}'
                    if self.args.usemeconv1x1_aftersptr:
                        self.args.exp_name += f'__localfc{self.args.beforeafter_convtype}'
                    if self.args.image_useswin:
                        self.args.exp_name += f'__swin{self.args.imageswin_type}'
                    else:
                        self.args.exp_name += f'__swinFalse'

                
        





        expr_dir = os.path.join(self.args.logdir, self.args.exp_name)
        self.args.results_dir = os.path.join(expr_dir, self.args.results_dir)
        self.args.models_dir = os.path.join(expr_dir, self.args.models_dir)
        self.args.runs_dir = os.path.join(expr_dir, self.args.runs_dir)
        mkdirs([self.args.logdir, expr_dir, self.args.runs_dir, self.args.models_dir, self.args.results_dir])

        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)


        return self.args


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)




if __name__ == '__main__':
    args = Options().parse()