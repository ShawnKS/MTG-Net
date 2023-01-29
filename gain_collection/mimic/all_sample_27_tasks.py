import sys,os
# sys.path.append('..')
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import *
from latent_patient_trajectories.representation_learner.run_model import *
import copy
from itertools import combinations
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--frac',type=int, default=0)
    parser.add_argument('--frac_num',type=int,default=200)
    parser.add_argument('--ratio',type=float,default = 0.1)
    args = parser.parse_args()
    return args

globalargs = get_args()

np.random.seed(globalargs.seed)
torch.manual_seed(globalargs.seed)
torch.cuda.manual_seed_all(globalargs.seed)
base_num = globalargs.frac*globalargs.frac_num
# my_sample_list = sample_list[globalargs.frac*globalargs.frac_num:globalargs.frac_num*(globalargs.frac+1)]
my_sample_list = [ ('icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous',
         'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy',
         'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_ill_defined','icd_injury', 'disch_24h', 'disch_48h', 'mort_24h', 'mort_48h',
         'Long LOS', 'Readmission 30', 'Final Acuity Outcome', 'dnr_24h', 'dnr_48h', 'cmo_24h', 'cmo_48h')]
# print(my_sample_list[0])
# sys.exit(0)
ratio = globalargs.ratio
loadargs = {
        'do_eicu': False, 
        # early_stopping applied?
        'early_stopping': True, 
                # available data ratio
        'ratio': ratio, 
                # which gpu to use
        'gpu': '1', 
        'do_train': True, 
        'do_eval_train': True, 
        'do_eval_tuning': True, 
        'early_stopping': True, 
        'batch_size': 512, 
        'do_eval_test': True, 
        'do_overwrite': True,
        # ~/Blob
        # home/covpreduser/Blob
        # mnt/blob_cache/
        'run_dir' : './comprehensive_MTL_EHR/sample_output/3000sample_27tasks/',
        'dataset_dir': './dataset/PTbenchmark/final_datasets_for_sharing/dataset/rotations/no_notes/0',}
args = Args.from_variable(loadargs)
datasets, train_dataloader = setup_for_run(args)
tuning_dataloader=None
if args.early_stopping:
# do random stopping and pass in tuning dataset
    tuning_dataloader=DataLoader(
            datasets['tuning'], sampler=RandomSampler(datasets['tuning']), batch_size=train_dataloader.batch_size,
            num_workers=1
        )
train_dataloader.dataset.set_epoch(0)
tuning_dataloader.dataset.set_epoch(0)
run_dir = './comprehensive_MTL_EHR/sample_output/3000sample_27tasks/seed_' + str(globalargs.seed)+ '_' + str(ratio) +'/'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
# for i in range(globalargs.frac_num):
select_tasks = ('icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous',
        'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy',
        'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_ill_defined','icd_injury', 'disch_24h', 'disch_48h', 'mort_24h', 'mort_48h',
        'Long LOS', 'Readmission 30', 'Final Acuity Outcome', 'dnr_24h', 'dnr_48h', 'cmo_24h', 'cmo_48h')
# select_tasks = my_sample_list[i]
# select_tasks = np.array(all_tasks)[random.sample(list(range(len(all_tasks))), i)]
ablate_tasks = [ 'icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous',
        'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy',
        'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_ill_defined','icd_injury', 'disch_24h', 'disch_48h', 'mort_24h', 'mort_48h',
        'Long LOS', 'Readmission 30', 'Final Acuity Outcome', 'dnr_24h', 'dnr_48h', 'cmo_24h', 'cmo_48h']
this_run_dir = run_dir + str( (base_num) ) + '_order_test/'
# if(os.path.exists(this_run_dir)):
#     continue
# python ./Scripts/all_sample_27_tasks.py --seed 1 --frac 0 --ratio
# 0.5 --frac_num 50
# else:
#     os.makedirs(this_run_dir)
for task in select_tasks:
    ablate_tasks.remove(task)
# ablate_tasks = []
Trainargs = {
        # sequence_len of input
        'max_seq_len': 48, 
        # 'modeltype' task-specific model after shared GRU backbone, only have FC temporally though we choose self_attention here :)
        'modeltype': 'self_attention', 
        # 'run_dir': '../Sample Args/920save/split1/0/0.1/acu1/', 
        # directory for saving
        'run_dir' : this_run_dir,
        'model_file_template': 'model',
        # should overwrite run dir?
        'do_overwrite': True,
        # which rotation(split)
        'rotation': 0, 
        # 'dataset_dir' Not used by default--inferred from rotation.
        'dataset_dir': './dataset/PTbenchmark/final_datasets_for_sharing/dataset/rotations/no_notes/0', 
        'num_dataloader_workers': 4, 
        'do_eicu': False, 
        # early_stopping applied?
        'early_stopping': True, 
        # available data ratio
        'ratio': ratio, 
        # which gpu to use
        'gpu': '1', 
        # maximum epochs
        'epochs': 100, 
        'do_train': True, 
        'do_eval_train': True, 
        'do_eval_tuning': True, 
        'do_eval_test': True, 
        'train_save_every': 1, 
        'batches_per_gradient': 1, 
        'set_to_eval_mode': '', 
        'notes': 'no_notes', 
        'do_train_note_bert': True, 
        # Model parameter settings
        'in_dim': 128, 'hidden_size': 128, 
        'intermediate_size': 128, 
        'num_attention_heads': 2, 
        'num_hidden_layers': 2, 
        'batch_size': 512, 
        'learning_rate': 0.0001, 
        'do_learning_rate_decay': True, 
        'learning_rate_decay': 1, 
        'learning_rate_step': 1, 
        'note_bert_lr_reduce': 1, 
        'kernel_sizes': [7, 7, 5, 3], 
        'num_filters': [10, 100, 100, 100], 
        'dropout': 0.5, 
        'gru_num_hidden': 2, 
        'gru_hidden_layer_size': 512, 
        'gru_pooling_method': 'last', 
        'task_weights_filepath': '', 
        'regression_task_weight': 0, 
        'do_add_cls_analog': False, 
        'do_masked_imputation': False, 
        'do_fake_masked_imputation_shape': False, 
        'imputation_mask_rate': 0, 
        'hidden_dropout_prob': 0.1, 
        'pooling_method': 'max', 
        'pooling_kernel_size': 4, 
        'pooling_stride': None, 
        'conv_layers_per_pool': 1, 
        'do_bidirectional': True, 
        'fc_layer_sizes': [256], 
        'do_weight_decay': True, 
        'weight_decay': 0, 
        'gru_fc_layer_sizes': [], 
        'ablate': ablate_tasks, 
        'balanced_race': False, 'do_test_run': False, 'do_detect_anomaly': False}
args = Args.from_variable(Trainargs)
args_run_setup(args)
print('start')
run_model(args, datasets, train_dataloader, tqdm=tqdm, tuning_dataloader=tuning_dataloader)
print('order',len(select_tasks))
sys.exit(0)
