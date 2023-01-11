import argparse, json, pickle
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, asdict

from ..constants import *

NOTE_OPTIONS = {'no_notes', 'no_notes_old', 'integrate_note_bert'}

class BaseArgs(ABC):
    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath, mode='r') as f: 
            return cls(**json.loads(f.read()))
    @staticmethod
    def from_pickle_file(filepath):
        with open(filepath, mode='rb') as f: return pickle.load(f)

    def to_dict(self): return asdict(self)
    def to_json_file(self, filepath):
        with open(filepath, mode='w') as f: f.write(json.dumps(asdict(self), indent=4))
    def to_pickle_file(self, filepath):
        with open(filepath, mode='wb') as f: pickle.dump(self, f)

    @classmethod
    @abstractmethod
    def _build_argparse_spec(cls, parser):
        raise NotImplementedError("Must overwrite in base class!")

    @classmethod
    def from_files(cls,run_dir):
        load_dir = run_dir
        args_filename = 'eval_args.json'
        args_path = os.path.join(load_dir, args_filename)
        # filename= eval_args.json
        new_args = cls.from_json_file(args_path)
        return new_args

    @classmethod
    def from_variable(cls,args):
        return cls(**args)

    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()
        # To load from a run_directory (not synced to overall structure above):
        parser.add_argument(
            "--do_load_from_dir", action='store_true',
            help="Should the system reload from the sentinel args.json file in the specified run directory "
                 "(--run_dir) and use those args rather than consider those set here? If so, no other args "
                 "need be set (they will all be ignored).",
            default=False
        )
        main_dir_arg, args_filename = cls._build_argparse_spec(parser)

        args = parser.parse_args()
        if args.do_load_from_dir:
            load_dir = vars(args)[main_dir_arg]
            assert os.path.exists(load_dir), "Dir (%s) must exist!" % load_dir
            args_path = os.path.join(load_dir, args_filename)
            assert os.path.exists(args_path), "Args file (%s) must exist!" % args_path
            new_args = cls.from_json_file(args_path)
            assert os.path.samefile(vars(new_args)[main_dir_arg], load_dir),\
                f"{main_dir_arg}: {vars(new_args)[main_dir_arg]} doesn't match loaded file: {load_dir}!"

            return new_args

        args_dict = vars(args)
        if 'do_load_from_dir' in args_dict: args_dict.pop('do_load_from_dir')
        return cls(**args_dict)

def intlt(bounds):
    start, end = bounds if type(bounds) is tuple else (0, bounds)
    def fntr(x):
        x = int(x)
        if x < start or x >= end: raise ValueError("%d must be in [%d, %d)" % (x, start, end))
        return x
    return fntr

def within(s):
    def fntr(x):
        if x not in s: raise ValueError("%s must be in {%s}!" % (x, ', '.join(s)))
        return x
    return fntr

# TODO(mmd): Validate args describe model.
# TODO(mmd): This class has a secret requirement that all fields be POD.
# TODO(mmd): VERY BRITTLE. MUST KEEP IN SYNC WITH ARGPARSE SPEC BELOW!
@dataclass
class Args(BaseArgs):
    # Configuration (do not change)
    max_seq_len:             int   = 48

    # Run Params (set)
    modeltype:               str   = "self_attention"
    run_dir:                 str   = "./tmp_output"
    model_file_template:     str   = "model" # can use {arg} format syntax
    do_overwrite:            bool  = False # should overwrite run dir?
    rotation:                int   = 0  # Must be in [0, 10)
    dataset_dir:             str   = None # Not used by default--inferred from rotation.
    num_dataloader_workers:  int   = 23 # Num dataloader workers. Can increase.

    do_eicu:                 bool  = False # This doesn't affect the dataset pulled, but rather affects model
                                           # dimensions and such.
    early_stopping:          bool = False
    ratio:                    float = 1
    gpu:                     str   = "0"
    # Training Params (set)
    epochs:                  int   = 2
    do_train:                bool  = True
    do_eval_train:           bool  = True
    do_eval_tuning:          bool  = True
    do_eval_test:            bool  = True
    train_save_every:        int   = 1
    batches_per_gradient:    int   = 1
    set_to_eval_mode:        str   = "" # if set, train under this eval mode.

    ## Early Stopping
    #patience:                int   = 5
    #do_early_stopping:          bool  = False
    #do_double_early_stopping:   bool  = False

    # Notes Training Params (set)
    notes:                   str   = "no_notes" # {no_notes, integrate_note_bert} TODO: add topics, doc2vec
    do_train_note_bert:      bool  = True
    #flip_note_bert_train:    int   = -1
    #custom_note_bert_config: bool  = False

    # Hyperparameters (tune)
    # DEPRECATED: in_dim
    in_dim:                  int   = 128 # all input features will be projected to this dimensionality.
    hidden_size:             int   = 128
    intermediate_size:       int   = 128
    num_attention_heads:     int   = 2
    num_hidden_layers:       int   = 2
    batch_size:              int   = 32
    learning_rate:           float = 1e-4
    # We track this separately to give hyperparameter tuning an easier way to disable this.
    do_learning_rate_decay:  bool  = True
    learning_rate_decay:     float = 1 # decay gamma. 1 is no change.
    learning_rate_step:      int   = 1
    note_bert_lr_reduce:     bool  = 1
    kernel_sizes:            tuple = (7, 7, 5, 3) # for CNN
    num_filters:             tuple = (10, 100, 100, 5) # last one must be 5
    dropout:                 float = 0.5 # dropout applied to final layer of CNN
    gru_num_hidden:          int   = 2
    gru_hidden_layer_size:   int   = 512
    gru_pooling_method:      str   = 'last'
    task_weights_filepath:   str   = "" # If empty, uses no weights.
    regression_task_weight:  float = 0 # By default Regression is OFF for now.
    do_add_cls_analog:       bool  = False
    do_masked_imputation:    bool  = False
    do_fake_masked_imputation_shape: bool  = False # For FT masked imputation models.
    imputation_mask_rate:    float = 0.0 # by default no masking.
    hidden_dropout_prob:     float = 0.1
    pooling_method:          str   = 'max'
    pooling_kernel_size:     int   = 4
    pooling_stride:          int   = None
    conv_layers_per_pool:    int   = 1
    do_bidirectional:        bool  = False
    fc_layer_sizes:          tuple = (256,)
    # We track this separately to give hyperparameter tuning an easier way to disable this.
    do_weight_decay:         bool  = True
    weight_decay:            float = 0
    gru_fc_layer_sizes:      tuple = tuple()

    ablate:                  tuple = tuple()
    reverse:                  tuple = tuple()

    frac_data:      float= 1.0 # how much of the fine_tuning data should we use?
    frac_data_seed: int  = 0 # how much of the fine_tuning data should we use?
    frac_female:    float= 1.0 # how much of the fine_tuning data should we use?
    frac_black:     float= 1.0 # how much of the fine_tuning data should we use?
    balanced_race:  bool = False # should the number of white patients be reduced to have class balance between black and white patients

    # Debug
    do_test_run:             bool  = False
    do_detect_anomaly:       bool  = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        # Configuration (do not change)
        parser.add_argument("--max_seq_len", type=int, default=48, help="maximum number of timepoints to feed into the model")
        parser.add_argument("--early_stopping",type = bool, default = False, help='if early_stopping is required')
        parser.add_argument("--ratio",type = float, default = 1, help='ratiooftrainingdata')
        # Run Params (set)
        parser.add_argument("--run_dir", type=str, required=True, help='save dir.')
        parser.add_argument("--do_overwrite", action='store_true', default=False, help='Should overwrite existent save_dir?')
        parser.add_argument("--no_do_overwrite", action='store_false', dest='do_overwrite')
        parser.add_argument('--rotation', type=intlt(10), default=0)
        parser.add_argument('--dataset_dir', type=str, default=None, help='Explicit dataset path (else use rotation).')
        parser.add_argument('--num_dataloader_workers', type=int, default=1, help='# dataloader workers.')
        parser.add_argument("--do_eicu", action="store_true", help="set flag to use eICU", default=False)
        parser.add_argument("--no_do_eicu", action="store_false", dest="do_eicu")
        parser.add_argument("--gpu", default = "0", help="gpu num")

        # Training Params (set)
        parser.add_argument("--modeltype", type=str, default='self_attention', choices = ['self_attention', 'cnn', 'gru', 'linear'], help="number of training epochs")
        parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
        parser.add_argument("--do_eval_train", action="store_true", help="set flag to train the model", default=True)
        parser.add_argument("--no_do_eval_train", action="store_false", dest="do_eval_train")
        parser.add_argument("--do_eval_tuning", action="store_true", help="set flag to val the model", default=True)
        parser.add_argument("--no_do_eval_tuning",   action="store_false", dest="do_eval_tuning")
        parser.add_argument("--do_eval_test", action="store_true", help="set flag to test the model", default=True)
        parser.add_argument("--no_do_eval_test",  action="store_false", dest="do_eval_test")
        parser.add_argument('--train_save_every', type=int, default = 1, help='Save the model every ? epochs?')
        parser.add_argument(
            '--batches_per_gradient', type=int, default = 1,
            help='Accumulate gradients over this many batches.'
        )
        parser.add_argument('--set_to_eval_mode', type=str, default="", help='train under this eval mode')

        # Early Stopping (Tune)
        #parser.add_argument('--patience', type=int, default = 10, help='Early stopping patience')
        #parser.add_argument('--do_early_stopping', action='store_true', default=False, help='do early stop?')
        #parser.add_argument('--no_do_early_stopping', action='store_false', dest='do_early_stopping')
        #parser.add_argument('--do_double_early_stopping', action="store_true", default=False)
        #parser.add_argument('--no_do_double_early_stopping', action="store_false", dest='do_double_early_stopping')

        # Notes Training Params (set)
        parser.add_argument('--notes', type=within(NOTE_OPTIONS), default='no_notes')
        parser.add_argument('--do_train_note_bert', action='store_true', default=True, help='Should note bert train?')
        parser.add_argument('--no_do_train_note_bert', action='store_false', dest='do_train_note_bert')
        #parser.add_argument('--flip_note_bert_train', type=int, default=0,help='if x = 0, no flipping. if x > 0, flip at epoch x. if x < 0, flip twice consecutively every x epochs')
        #parser.add_argument('--custom_note_bert_config', action="store_true", default=False)
        #parser.add_argument('--note_bert_hidden_dropout', type=float, default=0.1, help='Change the hidden layer dropout probability for note bert')

        # Hyperparameters (tune)
        parser.add_argument('--weight_decay', type=float, default=0, help="L2 weight decay penalty")
        parser.add_argument('--do_weight_decay', action='store_true', default=True, help="Do L2 weight decay?")
        parser.add_argument('--no_do_weight_decay', dest='do_weight_decay', action='store_false')
        parser.add_argument("--in_dim", type=int, default=128, help="input dimensionality")
        parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
        parser.add_argument("--intermediate_size", type=int, default=128, help="intermediate size")
        parser.add_argument("--num_attention_heads", type=int, default=2, help="# of attention heads")
        parser.add_argument("--num_hidden_layers", type=int, default=2, help="# of hidden layers")
        parser.add_argument("--batch_size", type=int, default=32, help="batch size for train, test, and eval")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for the model")
        parser.add_argument('--do_learning_rate_decay', action='store_true', default=True, help="Do learning rate decay?")
        parser.add_argument('--no_do_learning_rate_decay', dest='do_learning_rate_decay', action='store_false')
        parser.add_argument("--learning_rate_decay", type=float, default=1, help="lr decay factor")
        parser.add_argument("--learning_rate_step", type=int, default=1, help="#epochs / lr decay")
        parser.add_argument("--note_bert_lr_reduce", type=int, default=1, help='reduce the learning rate for note bert by this factor')
        parser.add_argument("--kernel_sizes", type=int, nargs='+', default=[7,7,5,3], help="filter sizes for CNN")
        parser.add_argument("--num_filters", type=int, nargs='+', default=[10,100,100,100], help="number of convolutional filters for top layers of cnn")
        parser.add_argument("--dropout", type=float, default=0.5, help="dropout applied to CNN")
        parser.add_argument("--gru_num_hidden", type=int, default=2, help="Number of hidden layers for GRU")
        parser.add_argument("--gru_hidden_layer_size", type=int, default=512, help="Hidden layer dimension for GRU.")
        parser.add_argument("--gru_pooling_method", type=str, default='last', help="GRU pooling style.")
        parser.add_argument(
            "--task_weights_filepath", type=str, default="", help="Path to json file of task weights."
        )
        parser.add_argument(
            '--do_add_cls_analog', action='store_true', default=False,
            help='Will add a [CLS] token analog for the transformer model. Only appropriate for transformers.'
        )
        parser.add_argument('--no_do_add_cls_analog', action='store_false', dest='do_add_cls_analog')

        parser.add_argument(
            '--do_masked_imputation', action='store_true', default=False,
            help=(
                'Will also run the masked imputation task. Likely only suitable for pre-training. '
                'Will not be applied during evaluation.'
            )
        )
        parser.add_argument('--no_do_masked_imputation', action='store_false', dest='do_masked_imputation')
        parser.add_argument(
            '--do_fake_masked_imputation_shape', action='store_true', default=False,
            help="Adds the mask dimension to the ts dim, for resuming from Masked Imputation models."
        )
        parser.add_argument('--no_do_fake_masked_imputation_shape', action='store_false', dest='do_fake_masked_imputation_shape')
        parser.add_argument('--imputation_mask_rate', type=float, default=0, help='Masking rate.')

        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='dropout')
        parser.add_argument('--pooling_method', type=within(('max', 'avg')), default='max', help='pooling?')
        parser.add_argument('--pooling_kernel_size', type=int, default=4, help='pooling kernel size')
        parser.add_argument('--pooling_stride', type=int, default=None, help='stride--None=same as kernel')
        parser.add_argument('--conv_layers_per_pool', type=int, default=1, help='conv layers / pool layer')
        parser.add_argument('--do_bidirectional', action='store_true', default=True, help='bidirectional?')
        parser.add_argument('--no_do_bidirectional', action='store_false', dest='do_bidirectional')
        parser.add_argument('--fc_layer_sizes', type=int, nargs='+', default=(256,), help='cnn fc stack')
        parser.add_argument('--gru_fc_layer_sizes', type=int, nargs='+', default=tuple(), help='gru fc stack')
        parser.add_argument('--regression_task_weight', type=float, default=0, help='weight on regresssion')
        parser.add_argument('--frac_data', type=float, default=1.0, help='# dataloader workers.')
        parser.add_argument('--frac_female', type=float, default=1.0, help='Number of females in terms of percent of male patients')
        parser.add_argument('--frac_black', type=float, default=1.0, help='Number of black pts vs. white')
        parser.add_argument('--frac_data_seed', type=int, default=0, help='random seed for subsampling the data for fine_tuning')
        parser.add_argument('--balanced_race', action='store_true',  help='Balance number of black and white patients during training?')

        # Debug
        parser.add_argument('--do_test_run', action='store_true', default=False, help='Will use small dataset. Faster runtime.')
        parser.add_argument('--no_do_test_run', action='store_false', dest='do_test_run')
        parser.add_argument('--do_detect_anomaly', action='store_true', default=False, help='Will detect nans. Slower runtime.')
        parser.add_argument('--no_do_detect_anomaly', action='store_false', dest='do_detect_anomaly')


        parser.add_argument(
            '--ablate', type=within(list(ABLATION_GROUPS.keys()) + ABLATION_ALL), nargs='+', default=tuple(),
            help='may include grouped tasks like icd10, discharge, mortality, los, readmission, '
                 'future_treatment_sequence, acuity, next_timepoint_info, dnr, or cmo. In addition, '
                 'individual tasks can be named.'
        )

        parser.add_argument(
            '--reverse', type=within(list(REVERSE_GROUPS.keys()) + ALL_TASKS), nargs='+', default=tuple(),
            help='may include grouped tasks like icd10, discharge, mortality, los, readmission, '
                 'future_treatment_sequence, acuity, next_timepoint_info, dnr, or cmo. In addition, '
                 'individual tasks can be named.'
        )
        return 'run_dir', ARGS_FILENAME

# TODO(mmd): Eventually make accept list of tasks so can work of groups of ablated tasks.
@dataclass
class FineTuneArgs(BaseArgs):
    run_dir:                  str   = "" # required
    do_eicu:                  bool  = False
    fine_tune_task:           str   = "" # required
    num_dataloader_workers:   int   = 8 # Num dataloader workers. Can increase.
    frac_fine_tune_data:      float = 1.0 # how much of the fine_tuning data should we use? # DEPRECATED
    frac_fine_tune_data_seed: int   = 0 # how much of the fine_tuning data should we use?
    frac_female:              float = 1.0 # how much of the fine_tuning data should we use?
    frac_black:               float = 1.0 # how much of the fine_tuning data should we use?
    balanced_race:            bool  = False
    train_embedding_after:    int   = -1 # should the embedding be frozen (-1) or trained after a number of epochs?
    do_match_train_windows:   bool  = True # Whether to match the train window regime to evaluation regime
    do_frozen_representation: bool  = True # Whether to train the FTD (fine-tuning, decoder-only) variant.
    do_free_representation:   bool  = False # Whether to train the FTF (fine-tuning, full) variant.
    do_small_data:            bool  = False # Whether to also train across various small-data levels
    do_single_task:           bool  = False
    do_masked_imputation_PT:  bool = False
    verbose:                  bool  = False # Whether to match the train window regime to evaluation regime

                                          # E.g., use only first 24 hours, etc.

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument(
            "--fine_tune_task", type=within(ALL_TASKS + list(ABLATION_GROUPS.keys())), help="Which task?"
        )
        parser.add_argument(
            '--do_masked_imputation_PT', action='store_true', default=True, help='Masked Imputation PT?'
        )
        parser.add_argument(
            '--no_do_masked_imputation_PT', action='store_false', dest='do_masked_imputation_PT'
        )
        parser.add_argument('--do_single_task', action='store_true', default=True, help='Single task?')
        parser.add_argument('--no_do_single_task', action='store_false', dest='do_single_task')
        parser.add_argument('--do_small_data', action='store_true', default=True, help='Small Data?')
        parser.add_argument('--no_do_small_data', action='store_false', dest='do_small_data')
        parser.add_argument('--do_frozen_representation', action='store_true', default=True, help='FTD?')
        parser.add_argument(
            '--no_do_frozen_representation', action='store_false', dest='do_frozen_representation'
        )
        parser.add_argument('--do_free_representation', action='store_true', default=False, help='FTE?')
        parser.add_argument(
            '--no_do_free_representation', action='store_false', dest='do_free_representation'
        )
        parser.add_argument("--do_eicu", action="store_true", help="set flag to use eICU", default=False)
        parser.add_argument("--no_do_eicu", action="store_false", dest="do_eicu")
        parser.add_argument("--run_dir", type=str, required=True, help="Dir for this generalizability exp.")
        parser.add_argument('--num_dataloader_workers', type=int, default=1, help='# dataloader workers.')
        parser.add_argument(
            '--do_match_train_windows', action='store_true', default=False,
            help='Matches the train windows used to the evaluation setting (e.g., use only the first 24 '
                 'hours, or only the last window_size hours, etc.'
        )
        parser.add_argument(
            '--no_do_match_train_windows', action='store_false', dest='do_match_train_windows'
        )
        parser.add_argument('--frac_fine_tune_data', type=float, default=1.0, help='# dataloader workers.')
        parser.add_argument('--frac_fine_tune_data_seed', type=int, default=0, help='random seed for subsampling the data for fine_tuning')
        parser.add_argument('--frac_female', type=float, default=1.0, help='# dataloader workers.')
        parser.add_argument('--frac_black', type=float, default=1.0, help='ratio of black patients compared to white patients')
        parser.add_argument('--balanced_race', action='store_true', help='Start with a balanced number of black patients to white patients.')

        parser.add_argument('--train_embedding_after', type=int, default=-1, help='Decide whether the embedding should be frozen (-1) or trained after this number of epochs. An argument of 0 will train the embedding for the whole fine-tuning window.')
        parser.add_argument('--verbose', action='store_true', help='print to motality file')

        return 'run_dir', FINE_TUNE_ARGS_FILENAME

@dataclass
class HyperparameterSearchArgs(BaseArgs):
    search_dir:    str  = "" # required
    algo:          str  = "tpe.suggest"
    max_evals:     int  = 10
    rotation:      int  = 1

    do_use_mongo:  bool = False
    mongo_addr:    str  = ""
    mongo_db:      str  = ""
    mongo_exp_key: str  = ""

    single_task_search: str = "" # By default is ignored.
    do_match_train_windows: bool = True # Match eval mode in training if single_task is set.

    do_eicu: bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument("--search_dir", type=str, required=True, help="Dir for this search process.")
        parser.add_argument("--algo", type=within({"tpe.suggest", "rand.suggest"}), default="tpe.suggest", help="Search algo")
        parser.add_argument("--max_evals", type=int, default=10, help="How many evals")
        parser.add_argument("--rotation", type=intlt(10), default=1, help="Rotation")

        # MongoDB (for parallel search)
        parser.add_argument('--do_use_mongo', action='store_true', default=False, help='Parallel via Mongo.')
        parser.add_argument('--no_do_use_mongo', action='store_false', dest='do_use_mongo')
        parser.add_argument("--mongo_addr", default="", type=str, help="Mongo DB Address for parallel search.")
        parser.add_argument("--mongo_db", default="", type=str, help="Mongo DB Name for parallel search.")
        parser.add_argument("--mongo_exp_key", default="", type=str, help="Mongo DB Experiment Key for parallel search.")

        parser.add_argument("--single_task_search", default="", type=str, help="Search over only a single task.")
        parser.add_argument(
            '--do_match_train_windows', action='store_true', default=False,
            help='Matches the train windows used to the evaluation setting (e.g., use only the first 24 '
                 'hours, or only the last window_size hours, etc.'
        )
        parser.add_argument(
            '--no_do_match_train_windows', action='store_false', dest='do_match_train_windows'
        )
        parser.add_argument(
            '--do_eicu', action='store_true', default=False, help='Run over the eICU dataset.'
        )
        parser.add_argument( '--no_do_eicu', action='store_false', dest='do_eicu')

        return 'search_dir', HYPERPARAMETER_SEARCH_ARGS_FILENAME

@dataclass
class TaskGeneralizabilityArgs(BaseArgs):
    exp_dir:                      str  = "" # required
    do_eicu:                      bool = False # This doesn't affect the dataset pulled, but rather affects model
    rotation:                     int  = 0
    do_eval:                      bool = True
    do_train:                     bool = True
    do_fine_tune:                 bool = True
    do_fine_tune_eval:            bool = True
    do_frozen_representation:     bool = True # Whether to train the FTD (fine-tuning, decoder-only) variant.
    do_free_representation:       bool = False # Whether to train the FTF (fine-tuning, full) variant.
    do_match_FT_train_windows:    bool = False # Whether to match the train window regime to evaluation regime in fine-tuning.
    slurm:                        bool = False
    partition:                    str  = 'p100'
    slurm_args:                   str  = ''
    do_small_data:                bool = False
    do_imbalanced_sex_data:       bool = False
    do_imbalanced_race_data:      bool = False
    train_embedding_after:        int  = -1 # should the embedding be frozen (-1) or trained after a number of epochs?
    do_single_task:               bool = False
    single_task:                  str  = None
    do_masked_imputation_PT:      bool = False
    do_copy_masked_imputation_PT: bool = True

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument("--exp_dir", type=str, required=True, help="Dir for this generalizability exp.")
        parser.add_argument(
            '--do_masked_imputation_PT', action='store_true', default=True, help='Masked Imputation PT?'
        )
        parser.add_argument(
            '--no_do_masked_imputation_PT', action='store_false', dest='do_masked_imputation_PT'
        )
        parser.add_argument(
            '--do_copy_masked_imputation_PT', action='store_true', default=True,
            help='Copy Masked Imputation PT?'
        )
        parser.add_argument(
            '--no_do_copy_masked_imputation_PT', action='store_false', dest='do_copy_masked_imputation_PT'
        )
        parser.add_argument('--do_single_task', action='store_true', default=True, help='Single task?')
        parser.add_argument('--no_do_single_task', action='store_false', dest='do_single_task')
        parser.add_argument('--single_task', type=str, default=None, help="What single task to do (if do)?")
        parser.add_argument("--do_eicu", action="store_true", help="set flag to use eICU", default=False)
        parser.add_argument("--no_do_eicu", action="store_false", dest="do_eicu")
        parser.add_argument("--rotation", type=intlt(10), default=0, help="Rotation")
        parser.add_argument('--do_eval', action='store_true', default=True, help='Evaluate as well?')
        parser.add_argument('--no_do_eval', action='store_false', dest='do_eval')
        parser.add_argument('--do_train', action='store_true', default=True, help='Train as well?')
        parser.add_argument('--no_do_train', action='store_false', dest='do_train')
        parser.add_argument('--do_frozen_representation', action='store_true', default=True, help='FTD?')
        parser.add_argument(
            '--no_do_frozen_representation', action='store_false', dest='do_frozen_representation'
        )
        parser.add_argument('--do_free_representation', action='store_true', default=False, help='FTE?')
        parser.add_argument(
            '--no_do_free_representation', action='store_false', dest='do_free_representation'
        )
        parser.add_argument('--do_fine_tune', action='store_true', default=True, help='Fine tune as well?')
        parser.add_argument('--no_do_fine_tune', action='store_false', dest='do_fine_tune')
        parser.add_argument('--do_fine_tune_eval', action='store_true', default=True, help='FT eval as well?')
        parser.add_argument('--no_do_fine_tune_eval', action='store_false', dest='do_fine_tune_eval')
        parser.add_argument(
            '--do_match_FT_train_windows', action='store_true', default=False,
            help='Matches the train windows used to the evaluation setting (e.g., use only the first 24 '
                 'hours, or only the last window_size hours, etc.) during fine tuning. Only used if '
                 'do_fine_tune is True.'
        )
        parser.add_argument(
            '--no_do_match_FT_train_windows', action='store_false', dest='do_match_FT_train_windows'
        )
        parser.add_argument(
            '--slurm', action='store_true', help='Just makes bash scripts for slurm'
        )
        parser.add_argument(
            '--partition', default='p100', help='slurm partition to launch job on'
        )
        parser.add_argument(
            '--slurm_args', default="", nargs='+', help='slurm args to add to the job. IE. #SBATCH --{" --".join(args.slurm_args[])}'
        )

        parser.add_argument(
            '--do_small_data', action='store_true', help='Include restricted tasks in fine-tuning?'
        )
        parser.add_argument(
            '--do_imbalanced_sex_data', action='store_true', help='Include frac female for fine-tuning?'
        )
        parser.add_argument(
            '--do_imbalanced_race_data', action='store_true', help='Include frac female for fine-tuning?'
        )
        parser.add_argument(
            '--train_embedding_after', type=int, default=-1, help='Decide whether the embedding should be frozen (-1) or trained after this number of epochs. An argument of 0 will train the embedding for the whole fine-tuning window.')

        return 'exp_dir', TASK_GEN_EXP_ARGS_FILENAME

@dataclass
class EvalArgs(BaseArgs):
    run_dir:                  str  = "" # required
    # notes and rotation need to be set properly, but are actually deprecated.
    notes:                    str  = "no_notes" # {no_notes, integrate_note_bert} TODO: add topics, doc2vec
    rotation:                 int  = 0
    dataset_dir:              str  = None # by default this is inferred
    do_eicu:                 bool  = False # This doesn't affect the dataset pulled, but rather affects model
    gpu:                     str   = "0"
    do_save_all_reprs:        bool = True
    do_eval_train:            bool = False
    do_eval_tuning:           bool = True
    do_eval_test:             bool = True
    num_dataloader_workers:   int  = 8 # Num dataloader workers. Can increase.

    do_debug_run:             bool = False
    do_overwrite:             bool = False
    eval_type:                str  = ''
    eval_epoch:               str  = 'latest'
    num_random_endpoints:     int  = 10

    do_masked_imputation:    bool  = False
    imputation_mask_rate:    float = 0.0 # by default no masking.

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument("--run_dir", type=str, required=True, help="Dir for this generalizability exp.")
        parser.add_argument("--rotation", type=intlt(10), default=0, help="Rotation")
        parser.add_argument('--dataset_dir', type=str, default=None, help='Explicit dataset path (else use rotation).')
        parser.add_argument("--gpu", default = "0", help="gpu num")
        parser.add_argument("--do_eicu", action="store_true", help="set flag to use eICU", default=False)
        parser.add_argument("--no_do_eicu", action="store_false", dest="do_eicu")
        parser.add_argument('--do_save_all_reprs', action='store_true', default=True, help='Save all reprs.')
        parser.add_argument('--no_do_save_all_reprs', action='store_false', dest='do_save_all_reprs')
        parser.add_argument('--do_eval_train', action='store_true', default=False, help='Eval Train')
        parser.add_argument('--no_do_eval_train', action='store_false', dest='do_eval_train')
        parser.add_argument('--do_eval_tuning', action='store_true', default=True, help='Eval Tuning')
        parser.add_argument('--no_do_eval_tuning', action='store_false', dest='do_eval_tuning')
        parser.add_argument('--do_eval_test', action='store_true', default=True, help='Eval Test')
        parser.add_argument('--no_do_eval_test', action='store_false', dest='do_eval_test')
        parser.add_argument('--notes', type=within(NOTE_OPTIONS), default='no_notes')
        parser.add_argument('--num_dataloader_workers', type=int, default=1, help='# dataloader workers.')

        parser.add_argument('--do_debug_run', action='store_true', default=False, help='Do a Debug Run')
        parser.add_argument('--no_do_debug_run', action='store_false', dest='do_debug_run')
        parser.add_argument('--do_overwrite', action='store_true', default=False, help='Overwrite any prior')
        parser.add_argument('--no_do_overwrite', action='store_false', dest='do_overwrite')
        parser.add_argument('--eval_type', type=str, default='', choices=['', 'female', 'male', 'black', 'white'], help='Evaluate the dataset on a subset of the population')
        parser.add_argument('--eval_epoch', default='latest', help='The epoch to evaluate')
        # parser.add_argument('--eval_metrics', type = str , help='which metrics to evaluate')
        parser.add_argument('--num_random_endpoints', type=int, default=10, help='How many random endpoints?')

        parser.add_argument(
            '--do_masked_imputation', action='store_true', default=False,
            help=(
                'Will also run the masked imputation task. Likely only suitable for pre-training. '
                'Will not be applied during evaluation.'
            )
        )
        parser.add_argument('--no_do_masked_imputation', action='store_false', dest='do_masked_imputation')
        parser.add_argument('--imputation_mask_rate', type=float, default=0, help='Masking rate.')

        return 'run_dir', EVAL_ARGS_FILENAME

@dataclass
class GetFlatReprArgs(BaseArgs):
    dataset_dir:              str  = "" # if empty, is ignored.
    notes:                    str  = "no_notes" # {no_notes, integrate_note_bert} TODO: add topics, doc2vec
    rotation:                 int  = 0

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--dataset_dir', type=str, default=None, help='Explicit dataset path (else use rotation).')
        parser.add_argument("--rotation", type=intlt(10), default=0, help="Rotation")
        parser.add_argument('--notes', type=within(NOTE_OPTIONS), default='no_notes')

        return 'dataset_dir', GET_ALL_FLAT_REPR_ARGS_FILENAME

@dataclass
class GetPCAArgs(BaseArgs):
    dataset_dir:    str   = "" # required
    train_filename: str   = "" # required
    eval_filenames: tuple = tuple()
    num_components: int   = 25
    open_mode:      str   = "pd_hdf"

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--dataset_dir', type=str, required=True, help='Explicit dataset path (else use rotation).')
        parser.add_argument('--train_filename', type=str, required=True, help='Train filename.')
        parser.add_argument('--eval_filenames', type=str, nargs='*', default=[], help='Eval filenames.')
        parser.add_argument('--num_components', type=int, default=25, help='How many components.')
        parser.add_argument('--open_mode', type=within({'pd_hdf','eval_reprs'}), default='pd_hdf', help='fmt')

        return 'dataset_dir', GET_PCA_ARGS_FILENAME

@dataclass
class ClusteringArgs(BaseArgs):
    run_dir:         str = "" # required
    train_filename:  str = "" # required
    tuning_filename: str = "" # required
    test_filename:   str = "" # required
    open_mode:       str = "pd_hdf"
    algo:            str = "tpe.suggest"

    # Hyperparameter Tuning
    clustering_config_filename: str  = CLUSTERING_CONFIG_FILENAME
    max_evals:                  int  = 100
    do_use_mongo:               bool = False
    mongo_addr:                 str  = ""
    mongo_db:                   str  = ""
    mongo_exp_key:              str  = ""

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument("--run_dir", type=str, required=True, help="Dir for this generalizability exp.")
        parser.add_argument('--train_filename', type=str, required=True, help='Train filename.')
        parser.add_argument('--test_filename', type=str, required=True, help='Test filename.')
        parser.add_argument('--tuning_filename', type=str, required=True, help='Tuning filename.')
        parser.add_argument('--open_mode', type=within({'pd_hdf','eval_reprs'}), default='pd_hdf', help='fmt')
        parser.add_argument("--algo", type=within({"tpe.suggest"}), default="tpe.suggest", help="Search algo")

        parser.add_argument(
            '--clustering_config_filename', type=str, default=CLUSTERING_CONFIG_FILENAME,
            help='Config filename.'
        )
        parser.add_argument("--max_evals", type=int, default=100, help="How many evals")

        # MongoDB (for parallel search)
        parser.add_argument('--do_use_mongo', action='store_true', default=False, help='Parallel via Mongo.')
        parser.add_argument('--no_do_use_mongo', action='store_false', dest='do_use_mongo')
        parser.add_argument("--mongo_addr", default="", type=str, help="Mongo DB Address for parallel search.")
        parser.add_argument("--mongo_db", default="", type=str, help="Mongo DB Name for parallel search.")
        parser.add_argument("--mongo_exp_key", default="", type=str, help="Mongo DB Experiment Key for parallel search.")

        return 'run_dir', CLUSTERING_ARGS_FILENAME
