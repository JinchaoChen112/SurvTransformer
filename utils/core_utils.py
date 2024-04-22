from ast import Lambda
import numpy as np
import os
from models.model_SurvTransformer import SurvTransformer
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

import torch

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, UniLoss, l1_reg_modules

import torch.optim as optim

class EarlyStopping:
    def __init__(self, warmup=5, patience=10, stop_epoch=15, verbose=False):

        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class Monitor_CIndex:
    def __init__(self):

        self.best_score = None

    def __call__(self, epoch, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)

def _get_splits(datasets, cur, args):

    print('\nTraining Fold {}!'.format(cur+1))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split, val_split


def _init_loss_function(args):

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        UniLoss_fn = UniLoss()
    else:
        raise NotImplementedError
    print('Done!')

    if args.reg_type == 'pathcell':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    return loss_fn, UniLoss_fn, reg_fn


def _init_optim(args, model):

    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):

    print('\nInit Model...', end=' ')

    if args.modality == "SurvTransformer":
        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes, 'wsi_embedding_dim': args.encoding_dim}
        model = SurvTransformer(**model_dict)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader, val_loader

def _extract_survival_metadata(train_loader, val_loader):

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):

    if modality in ["SurvTransformer"]:
        data_WSI = data[0].to(device)

        data_omics = [[] for _ in range(len(data[1]))]
        for idx in range(len(data[1])):
            for item in data[1][idx]:
                data_omics[idx].append(item.to(device))
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list

def _process_data_and_forward(model, modality, device, data):

    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list = _unpack_data(modality, device, data)


    if modality == "SurvTransformer":

        input_args = {"x_path": data_WSI.to(device)}
        for idx in range(len(data_omics)):
            omic_list = {}
            for i in range(len(data_omics[idx])):
                omic_list['x_omic%s' % str(i + 1)] = data_omics[idx][i].type(torch.FloatTensor).to(device)
                input_args['batch%s' % str(idx + 1)] = omic_list
                
        input_args["return_attn"] = False
        input_args["contrast_loss"] = False

        h = model(**input_args)
        
    else:
        input_args = {"data_WSI": data_WSI.to(device),
                      "data_omics": data_omics.to(device)}
        h = model(**input_args)

    return h, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):

    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):

    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, loss_fn, UniLoss_fn, reg_fn=None, lambda_reg=0.):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    total_loss_reg = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            h_cw, h_cell, h_wsi, batch_omic_bag, wsi_embed = h
            loss = 0.5*loss_fn(h=h_cw, y=y_disc, t=event_time, c=censor)
            #loss_value = loss.item()
            loss += 0.25*loss_fn(h=h_cell, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_wsi, y=y_disc, t=event_time, c=censor)
            loss += 0.25*UniLoss_fn(modalA=batch_omic_bag, modalB=wsi_embed, logitsA=h_cell, logitsB=h_wsi, label=y_disc)
            loss_value = loss.item()
            h = h_cw

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value
        total_loss_reg += loss_value + loss_reg

        loss = loss / y_disc.shape[0] + loss_reg
        loss.backward()

        optimizer.step()

    total_loss /= len(loader.dataset)
    total_loss_reg /= len(loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_loss_reg: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, total_loss_reg, c_index))

    return c_index, total_loss


def validate_survival(dataset_factory, model, modality, loader, loss_fn, UniLoss_fn,
                      survival_train=None, early_stopping=None, monitor_cindex=None,
                      results_dir=None, epoch=None, cur=None, reg_fn=None, lambda_reg=0.):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.
    total_loss_reg = 0.
    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list = _unpack_data(modality,
                                                                                                            device,
                                                                                                            data)
            if modality == "SurvTransformer":

                input_args = {"x_path": data_WSI.to(device)}
                for idx in range(len(data_omics)):
                    omic_list = {}
                    for i in range(len(data_omics[idx])):
                        omic_list['x_omic%s' % str(i + 1)] = data_omics[idx][i].type(torch.FloatTensor).to(device)
                        input_args['batch%s' % str(idx + 1)] = omic_list
                input_args["return_attn"] = False
                input_args["contrast_loss"] = False

                h = model(**input_args)

            else:
                input_args = {"data_WSI": data_WSI.to(device),
                              "data_omics": data_omics.to(device)}
                h = model(**input_args)

            if not isinstance(h, tuple):
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()

            else:
                h_cw, h_cell, h_wsi, batch_omic_bag, wsi_embed = h
                loss = 0.5 * loss_fn(h=h_cw, y=y_disc, t=event_time, c=censor)
                #loss_value = loss.item()
                loss += 0.25 * loss_fn(h=h_cell, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_wsi, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * UniLoss_fn(batch_omic_bag, wsi_embed, h_cell, h_wsi, y_disc)
                loss_value = loss.item()
                h = h_cw

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores,
                                                                                                   all_censorships,
                                                                                                   all_event_times,
                                                                                                   all_clinical_data,
                                                                                                   event_time, censor,
                                                                                                   risk,
                                                                                                   clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            total_loss_reg += loss_value + loss_reg

            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    total_loss_reg /= len(loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores,
                                                          all_censorships, all_event_times, all_risk_by_bin_scores)
    print('val epoch:{},  c-index:{}'.format(epoch, c_index))

    if early_stopping:
        print('------start early_stopping------')
        assert results_dir
        early_stopping(epoch, total_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    if monitor_cindex:
        print('------start monitor_cindex------')
        assert results_dir
        monitor_cindex(epoch, c_index, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_maxC_index_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    
    data = loader.dataset.metadata["survival_months"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc

    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.

    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, UniLoss_fn, survival_train=None, reg_fn=None, lambda_reg=0.):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    total_loss_reg = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():

        for data in loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list = _unpack_data(modality, device, data)

            if modality == "SurvTransformer":

                input_args = {"x_path": data_WSI.to(device)}
                for idx in range(len(data_omics)):
                    omic_list = {}
                    for i in range(len(data_omics[idx])):
                        omic_list['x_omic%s' % str(i + 1)] = data_omics[idx][i].type(torch.FloatTensor).to(device)
                        input_args['batch%s' % str(idx + 1)] = omic_list

                input_args["return_attn"] = False
                input_args["contrast_loss"] = False

                h = model(**input_args)

            else:
                input_args = {"data_WSI": data_WSI.to(device),
                              "data_omics": data_omics.to(device)}
                h = model(**input_args)

            if not isinstance(h, tuple):
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()
            else:
                h_cw, h_cell, h_wsi, batch_omic_bag, wsi_embed = h
                loss = 0.5 * loss_fn(h=h_cw, y=y_disc, t=event_time, c=censor)
                #loss_value = loss.item()
                loss += 0.25 * loss_fn(h=h_cell, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * loss_fn(h=h_wsi, y=y_disc, t=event_time, c=censor)
                loss += 0.25 * UniLoss_fn(batch_omic_bag, wsi_embed, h_cell, h_wsi, y_disc)
                loss_value = loss.item()
                h = h_cw

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())

            total_loss += loss_value
            total_loss_reg += loss_value + loss_reg

            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    total_loss_reg /= len(loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss

def _step(cur, args, loss_fn, UniLoss_fn, model, optimizer, train_loader, val_loader, monitor_cindex, early_stopping, reg_fn):

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    
    if not args.need_test:
        for epoch in range(args.max_epochs):
            _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, loss_fn, UniLoss_fn, reg_fn, args.lambda_reg)
            stop = validate_survival(args.dataset_factory, model, args.modality,  val_loader, loss_fn, UniLoss_fn,
                                     all_survival, early_stopping, monitor_cindex, args.results_dir, epoch, cur,
                                     reg_fn, args.lambda_reg)

    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory,
                                                                                                model, args.modality,
                                                                                                val_loader, loss_fn,
                                                                                                UniLoss_fn, all_survival,
                                                                                                reg_fn, args.lambda_reg)
    
    print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
        val_cindex, 
        val_cindex_ipcw,
        val_IBS,
        val_iauc
        ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _train_val(datasets, cur, args):

    train_split, val_split = _get_splits(datasets, cur, args)

    loss_fn, UniLoss_fn, reg_fn = _init_loss_function(args)

    model = _init_model(args)

    optimizer = _init_optim(args, model)

    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    if args.early_stopping:
        monitor_cindex = Monitor_CIndex()
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None
        monitor_cindex = None

    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn,
                                                                                           UniLoss_fn,
                                                                                           model, optimizer,
                                                                                           train_loader, val_loader,
                                                                                           monitor_cindex, early_stopping,
                                                                                           reg_fn)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)