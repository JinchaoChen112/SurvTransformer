import os
import pandas as pd 

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import torch.nn.functional as F
import math
from itertools import islice
import collections
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _prepare_for_experiment(args):

    args.device = device
    print(args.device)
    args.split_dir = os.path.join("splits", args.which_splits, args.study)
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)

    assert os.path.isdir(args.split_dir)
    print('Split dir:', args.split_dir)

    _create_results_dir(args)

    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.study,
                'reg': args.reg,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "num_patches":args.num_patches,
                'split_dir': args.split_dir,
                'fusion':args.fusion,
                'modality':args.modality
                }

    _print_and_log_experiment(args, settings)

    composition_df = pd.read_csv("./datasets_csv/{}/cell_signatures/{}_cell_sparse_matrix.csv".format(args.study, args.study),
                encoding='latin1', index_col=0)
    args.composition_df = composition_df

    return args

def _print_and_log_experiment(args, settings):

    with open(args.results_dir + '/experiment_{}.txt'.format(args.param_code), 'w') as f:
        print(settings, file=f)

    f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")



def _get_custom_exp_code(args):

    param_code = ''

    param_code += args.study + "_"

    param_code += '_%s' % args.bag_loss

    param_code += '_lr%s' % format(args.lr, '.0e')


    param_code += '_%s' % args.which_splits.split("_")[0]

    param_code += '_b%s' % str(args.batch_size)

    param_code += "_" + args.label_col

    param_code += "_patches_" + str(args.num_patches)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim)
    param_code += "_epochs_" + str(args.max_epochs)
    param_code += "_modality_" + str(args.modality)

    args.param_code = param_code

    return args


def _seed_torch(seed=8):

    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _create_results_dir(args):

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

def _get_start_end(args):

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)
    return folds

def _save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].metadata['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])

    df.to_csv(filename)
    print()


def _series_intersection(s1, s2):
    return pd.Series(list(set(s1) & set(s2)))

def _print_network(results_dir, net):

    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    fname = "model_" + results_dir.split("/")[-1] + ".txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write('Total number of parameters: %d \n' % num_params)
    f.write('Total number of trainable parameters: %d \n' % num_params_train)
    f.close()


def _collate_SurvTransformer(batch):
    img = torch.stack([item[0] for item in batch])

    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omic_data_list, label, event_time, c, clinical_data_list, mask]


def _make_weights_for_balanced_classes_split(dataset):

    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                   
        weight[idx] = weight_per_class[y]   

    final_weights = torch.DoubleTensor(weight)

    return final_weights

class SubsetSequentialSampler(Sampler):

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)


def _get_split_loader(args, split_dataset, training = False, testing = False, weighted = False, batch_size=1):

    kwargs = {'num_workers': 0} if device.type == "cuda" else {}

    if args.modality == "SurvTransformer":
        collate_fn = _collate_SurvTransformer
    else:
        raise NotImplementedError

    if not testing:
        if training:
            if weighted:
                weights = _make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, drop_last=False, **kwargs)	
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, drop_last=False, **kwargs )

    return loader
