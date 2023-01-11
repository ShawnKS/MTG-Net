"""
run_model.py
"""
# python run_model.py --do_load_from_dir --run_dir ../Sample\ Args/823save/icdwbm/

import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import time
import json, os, pickle
from tqdm import tqdm

# TODO: check these imports.
from ..utils import *
from ..constants import *
from ..data_utils import *
from ..representation_learner.fts_decoder import *
from ..representation_learner.evaluator import *
from ..representation_learner.meta_model import *
from ..BERT.model import *
from ..BERT.constants import *
from .args import Args, EvalArgs

def train_meta_model(
    meta_model, train_dataloader, args, reloaded=False, epoch=0,
    tuning_dataloader=None, train_embedding_after=-1, tqdm=tqdm, just_gen_data=False,
):
    print(meta_model.parameters)
    print(sum(p.numel() for p in meta_model.parameters if p.requires_grad))
    for p in meta_model.parameters:
        if p.requires_grad:
            print(p.shape)
    sys.exit(0)
    all_train_perfs, all_dev_perfs = [], []
    if just_gen_data:
        optimizer, scheduler = None, None
    else:
        optimizer = torch.optim.Adam(
            meta_model.parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay if args.do_weight_decay else 0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.learning_rate_step,
            args.learning_rate_decay if args.do_learning_rate_decay else 1,
        )

    early_stop_count=0
    prev_err=10e9

    epoch_rng = range(epoch+1 if reloaded else 0, args.epochs)
    # train_dataloader.dataset.set_epoch(0)
    # tuning_dataloader.dataset.set_epoch(0)
    if tqdm is not None: epoch_rng = tqdm(epoch_rng, desc='Epoch: N/A', leave=False)
# 这里对应时间表长度
    for epoch in epoch_rng:
        if not just_gen_data:
            scheduler.step() # This goes before any of the train/validation stuff b/c torch version is 1.0.1

            if train_embedding_after >= epoch:
                # to ensure it is unfrozen after reloading
                meta_model.unfreeze_representation()

            meta_model.train()
            optimizer.zero_grad()
        dataloader_rng = train_dataloader
        if tqdm is not None:
            dataloader_rng = tqdm(
                dataloader_rng, desc='Batch: N/A', total=len(train_dataloader), leave=False)
        batchtime = time.time()
        for i, batch in enumerate(dataloader_rng):
            # print("load_time is ", time.time()-batchtime)
            if just_gen_data:
                total_loss = torch.tensor(0)
                continue
            if args.do_detect_anomaly: set_detect_anomaly(True)

            if batch['ts'].shape[0] == 1:
                print("Skipping singleton batch.")
                continue
            tmptime = time.time()
            hidden_states, pooled_output, all_outputs, total_loss = meta_model.forward(batch)
            # print("forward time is", time.time() - tmptime)
            try:
                tmptime = time.time()
                total_loss.backward()
                # print("loss_backward time is ",time.time() - tmptime )
            except:
                # print(total_loss.shape, total_loss)
                raise
            if i % args.batches_per_gradient == 0:
                optimizer.step()
                optimizer.zero_grad()
            if args.do_detect_anomaly: set_detect_anomaly(False)

            # if tqdm is not None: dataloader_rng.set_description('Batch: %.2e' % total_loss)
            batchtime = time.time()

        if just_gen_data: continue

        # if tqdm is None: print("Epoch %d: %.2f" % (epoch, total_loss.item()))
        # elif (tqdm is not None) and (tuning_dataloader is not None): pass
        # else: epoch_rng.set_description("Epoch %d: %.2f" % (epoch, total_loss.item()))
        if epoch % args.train_save_every == 0:
            if tuning_dataloader is None:
                meta_model.save(epoch)
            else:
                # print("Doing early stop")
                # do eval to see if this is the best score
                # tuning_dataloader.dataset.epoch= epoch
                meta_model.eval()
                dataloader_rng = tuning_dataloader
                if tqdm is not None:
                    dataloader_rng = tqdm(
                        dataloader_rng, desc='Batch: N/A', total=len(tuning_dataloader), leave=False)
                tuning_losses=[]
                for i, batch in enumerate(dataloader_rng):
                    hidden_states, pooled_output, all_outputs, evaluate_multitotal_loss = meta_model.forward(batch,tuning=True)
                    tuning_losses.append(evaluate_multitotal_loss.cpu().data.numpy().ravel())
                meta_model.train()
                total_err = np.mean(np.concatenate(tuning_losses))
                if(epoch == 0):
                    best_meta_model = meta_model
                if total_err < prev_err:
                    # this is the best model
                    meta_model.save(epoch)
                    best_meta_model = meta_model
#                     best_meta_model=meta_model.copy().cpu()
                    prev_err=total_err
                    early_stop_count=0
                    # if tqdm is None: print("Epoch %d: %.2f" % (epoch, evaluate_multitotal_loss.item()))
                    # else: epoch_rng.set_description("Epoch %d: %.2f" % (epoch, evaluate_multitotal_loss.item()))
                else:
                    meta_model.save(epoch)
                    early_stop_count+=1
                    if early_stop_count==15:
                        print(f"Early stopping at epoch {epoch}. Best model at epoch {epoch} with a loss of {prev_err}")
                        save_path = os.path.join(args.run_dir, 'optimizer_state_epoch-%d' % epoch)
                        torch.save(optimizer.state_dict(),save_path)
                        break


    if just_gen_data: return None

    if tuning_dataloader is None:
        meta_model.save(epoch)
        return meta_model
    else: 
        return best_meta_model

def run_model(
    args, datasets, train_dataloader, tqdm=None, meta_model=None, tuning_dataloader=None, just_gen_data=False
):
    # 默认meta_model=None
    if just_gen_data:
        meta_model = None
        reloaded, epoch = None, 0
    else:
        if meta_model is None: meta_model = MetaModel(
            args, datasets['train'][0],
            class_names = {'tasks_binary_multilabel': datasets['train'].get_binary_multilabel_keys()}
        )
        reloaded, epoch = meta_model.load()
        if reloaded: print("Resuming from epoch %d" % (epoch+1))
    if args.do_train:
        print('training')
        train_meta_model(
            meta_model, train_dataloader, args, reloaded, epoch, tuning_dataloader, tqdm=tqdm,
            just_gen_data=just_gen_data
        )

    #for n, do, dataloader in (
    #    ('train', args.record_train_perf, train_dataloader), ('dev', args.val, dev_dataloader),
    #    ('test', args.test, test_dataloader)
    #):
    #    if not do: continue
    #    perf_dict = eval_loop(
    #        model, device, dataloader, tqdm=tqdm, run_dir=args.run_dir, n_gpu=n_gpu, ablations=args.ablate,
    #        integrate_note_bert=args.integrate_note_bert, note_embedding_model=note_embedding_model, notes_projector=notes_projector,
    #        batch_size=args.batch_size, only_notes=args.only_notes, train_note_bert=args.train_note_bert,
    #        using_notes_embeddings=using_notes_embeddings
    #    )
    #    with open(os.path.join(args.run_dir, '{}.eval'.format(n)), mode='wb') as f:
    #        pickle.dump(perf_dict, f)
    return meta_model

def load_datasets(
    args, just_gen_data=False, use_stored_epochs=False, use_dataset_shells=True, make_train_dataloader=True
):
    do_splits_dict = {}
    if type(args) is Args:
        if not hasattr(args, 'dataset_dir') or not args.dataset_dir:
            rotations_dir = EICU_ROTATIONS_DIR if args.do_eicu else ROTATIONS_DIR
            args.dataset_dir = os.path.join(rotations_dir, args.notes, str(args.rotation))

        do_splits_dict['train']  = args.do_train or args.do_eval_train
        do_splits_dict['tuning'] = args.do_eval_tuning
        do_splits_dict['test']   = args.do_eval_test
        max_seq_len              = args.max_seq_len
        set_to_eval_mode         = args.set_to_eval_mode
    elif type(args) is EvalArgs:
        print(args)
        print(args.run_dir)
        print(ARGS_FILENAME)
        print(os.path.join(args.run_dir, ARGS_FILENAME))
        training_args = Args.from_json_file(os.path.join(args.run_dir, ARGS_FILENAME))
        for arg in ('notes', 'rotation',  'imputation_mask_rate', 'do_masked_imputation'):
            if hasattr(args, arg) and getattr(args, arg) not in (None, ''):
                assert hasattr(training_args, arg)
                assert getattr(training_args, arg) == getattr(args, arg), \
                    f"Dataset parameters disagree ({arg})!"
        if not hasattr(args, 'dataset_dir') or not args.dataset_dir:
            if hasattr(training_args, 'dataset_dir'): 
                args.dataset_dir = training_args.dataset_dir
            else:
                rotations_dir = EICU_ROTATIONS_DIR if args.do_eicu else ROTATIONS_DIR
                args.dataset_dir = os.path.join(
                    rotations_dir, training_args.notes, str(training_args.rotation)
                )

        do_splits_dict['train']  = args.do_eval_train
        do_splits_dict['tuning'] = args.do_eval_tuning
        do_splits_dict['test']   = args.do_eval_test
        max_seq_len              = training_args.max_seq_len
        set_to_eval_mode         = EVAL_MODES[1] # "first_24"
    else: raise AssertionError(f"Args must be of a recognized type! Is {type(args)}.")

    datasets = {}
    for split, do in do_splits_dict.items():
        if not do:
            datasets[split] = None
            continue

        load_start = time.time()
        dataset_shell_path = os.path.join(args.dataset_dir, f"{split}_dataset_shell.pkl")
        if use_dataset_shells and os.path.isfile(dataset_shell_path): load_path = dataset_shell_path
        else: load_path = os.path.join(args.dataset_dir, f"{split}_dataset.pkl")
        datasets[split] = depickle(load_path)

        if hasattr(datasets[split], 'skip_cache') and datasets[split].skip_cache:
            print(f"{load_path} has skip_cache true.")

        datasets[split].skip_cache = False
        if just_gen_data: datasets[split].save_data_only = True
        else: datasets[split].save_data_only = False
        print('loading %s data from disk took %.2f minutes' % (split, (time.time() - load_start)/60))

        datasets[split].reload_self_dir = args.dataset_dir
        datasets[split].train_tune_test = split
        datasets[split].save_place = os.path.join(
            args.dataset_dir, "stored_epochs" if use_stored_epochs else "stored_items"
        )
        if split == 'train':
            datasets[split].max_seq_len = max_seq_len

        if set_to_eval_mode: datasets[split].set_to_eval_mode(set_to_eval_mode)
        elif split != 'train': datasets[split].set_to_eval_mode(EVAL_MODES[1])

        datasets[split].set_epoch(0)
        if not os.path.isdir(datasets[split].save_place): os.makedirs(datasets[split].save_place)
        if args.do_masked_imputation:
            assert args.imputation_mask_rate > 0, "Can't do imputation masking if we mask nothing!"
            datasets[split].imputation_mask_rate = args.imputation_mask_rate
        else:
            assert args.imputation_mask_rate == 0, "Can't mask if imputation masking is not enabled."
            assert datasets[split].imputation_mask_rate == 0, "Shouldn't mask!" 
    # print(datasets['train'][0])
    # sys.exit(0)
    return datasets

def args_run_setup(args):
    # Make run_dir if it doesn't exist.
    if not os.path.exists(args.run_dir): 
        os.mkdir(os.path.abspath(args.run_dir))
    elif not args.do_overwrite:
        raise ValueError("Save dir %s exists and overwrite is not enabled!" % args.run_dir)

    if not args.dataset_dir:
        rotations_dir = EICU_ROTATIONS_DIR if args.do_eicu else ROTATIONS_DIR
        args.dataset_dir = os.path.join(rotations_dir, args.notes, str(args.rotation))
# rotations_dir = run_dir
    args.to_json_file(os.path.join(args.run_dir, ARGS_FILENAME))
    return

def setup_datasets_and_dataloaders(args, just_gen_data=False, use_stored_epochs=False):
    datasets = load_datasets(
        args, just_gen_data=just_gen_data, use_stored_epochs=use_stored_epochs
    )

    if not args.do_train: return datasets
    ratio = args.ratio
    sampler = SubsetRandomSampler(list(range(50))) if args.do_test_run else RandomSampler(datasets['train'])
    # 加一个 few-shot sampler
    sampler = SubsetRandomSampler(list( range( (int)(ratio * len(datasets['train']) ) ) ))
    # sampler is random!
    if just_gen_data:
        # In this case we override the typical collate_fn to avoid any errors with partially read and
        # partially constructed keys.
        train_dataloader = DataLoader(
            datasets['train'], sampler=sampler, batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers, collate_fn=lambda xs: dict(),
        )
    else:
        train_dataloader = DataLoader(
            datasets['train'], sampler=sampler, batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers
        )

    return datasets, train_dataloader

def setup_for_run(args, just_gen_data=False, use_stored_epochs=False):
    args_run_setup(args)
    return setup_datasets_and_dataloaders(
        args, just_gen_data=just_gen_data, use_stored_epochs=use_stored_epochs
    )

def main(args, tqdm):
    datasets, train_dataloader = setup_for_run(args)
    tuning_dataloader=None
    if args.early_stopping:
        # do random stopping and pass in tuning dataset
        tuning_dataloader=DataLoader(
                datasets['tuning'], sampler=RandomSampler(datasets['tuning']), batch_size=train_dataloader.batch_size,
                num_workers=1
            )
    return run_model(args, datasets, train_dataloader, tqdm=tqdm, tuning_dataloader=tuning_dataloader)
