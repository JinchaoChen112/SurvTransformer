from __future__ import print_function, division
from cProfile import label
import os
import pdb
from unittest import case
import pandas as pd

import pickle
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.general_utils import _series_intersection


class SurvivalDatasetFactory:

    def __init__(self,
        study,
        sig,
        seed, 
        print_info, 
        n_bins, 
        label_col, 
        eps=1e-6,
        num_patches=4096,
        is_survtransformer=True
        ):
        self.study = study
        self.sig = sig
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids  = (None, None)
        self.data_dir = None
        self.label_col = label_col
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.is_survtransformer = is_survtransformer

        self.label_col == "survival_months"
        self.survival_endpoint = "OS"
        self.censorship_var = "censorship"

        self._setup_omics_data() 

        self._setup_metadata_and_labels(eps)

        self._cls_ids_prep()

        self._load_clinical_data()

        self._summarize()

        if self.is_survtransformer:
            self._setup_survtransformer()
        else:
            self.omic_names = []
            self.omic_sizes = []

    def _setup_survtransformer(self):

        self.signatures = pd.read_csv("./datasets_csv/{}/cell_signatures/{}_{}_cell_signatures.csv".format(self.study, self.sig, self.study),
                                     encoding='latin1')
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]
            

    def _load_clinical_data(self):

        path_to_data = "./datasets_csv/{}/clinical_data/{}_clinical.csv".format(self.study, self.study)
        self.clinical_data = pd.read_csv(path_to_data, index_col=0)
    
    def _setup_omics_data(self):

        self.all_modalities = {}
        self.all_modalities['rna'] = pd.read_csv(
                './datasets_csv/{}/rna_data/{}_{}_rna_clean.csv'.format(self.study, self.sig, self.study),
                engine='python',
                index_col=0
            )

    def _setup_metadata_and_labels(self, eps):

        self.label_data = pd.read_csv('datasets_csv/{}/clinical_data/{}_os_slide.csv'.format(self.study, self.study))

        uncensored_df = self._clean_label_data()

        self._discretize_survival_months(eps, uncensored_df)

        self._get_patient_dict()

        self._get_label_dict()

        self._get_patient_data()

    def _clean_label_data(self):

        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df = self.patients_df[self.patients_df[self.censorship_var] < 1]
        
        return uncensored_df

    def _discretize_survival_months(self, eps, uncensored_df):

        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps

        disc_labels, q_bins = pd.cut(self.patients_df[self.label_col], bins=q_bins, retbins=True,
                                     labels=False, right=False, include_lowest=True)
        self.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins
        
    def _get_patient_data(self):

        patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')] 
        patient_data = {'case_id': patients_df["case_id"].values,
                        'label': patients_df['label'].values}
        self.patient_data = patient_data

    def _get_label_dict(self):

        label_dict = {}
        key_count = 0
        for i in range(len(self.bins)-1):
            for c in [0, 1]:
                label_dict.update({(i, c):key_count})
                key_count+=1

        for i in self.label_data.index:
            key = self.label_data.loc[i, 'label']
            self.label_data.at[i, 'disc_label'] = key
            censorship = self.label_data.loc[i, self.censorship_var]
            key = (key, int(censorship))
            self.label_data.at[i, 'label'] = label_dict[key]

        self.num_classes=len(label_dict)
        self.label_dict = label_dict

    def _get_patient_dict(self):
        patient_dict = {}
        temp_label_data = self.label_data.set_index('case_id')
        for patient in self.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        self.patient_dict = patient_dict
        self.label_data = self.patients_df
        self.label_data.reset_index(drop=True, inplace=True)

    def _cls_ids_prep(self):

        self.patient_cls_ids = [[] for i in range(self.num_classes)]   

        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 


        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.label_data['label'] == i)[0]

    def _summarize(self):

        if self.print_info:
            print("label column: {}".format(self.label_col))
            print("number of cases {}".format(len(self.label_data)))
            print("number of classes: {}".format(self.num_classes))

    def _patient_data_prep(self):
        patients = np.unique(np.array(self.label_data['case_id']))
        patient_labels = []
        
        for p in patients:
            locations = self.label_data[self.label_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.label_data['label'][locations[0]]
            patient_labels.append(label)
        
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        _, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def return_splits(self, args, csv_path, fold):

        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        print("Defining datasets...")
        train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)
        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val', fold=fold, scaler=scaler)

        args.omic_sizes = args.dataset_factory.omic_sizes
        datasets = (train_split, val_split)
        
        return datasets

    def _get_scaler(self, data):

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler
    
    def _apply_scaler(self, data, scaler):

        zero_mask = data == 0

        transformed = scaler.transform(data)
        data = transformed

        data[zero_mask] = 0.
        
        return data

    def _get_split_from_df(self, args, all_splits, split_key: str='train', fold = None, scaler=None, valid_cols=None):

        if not scaler:
            scaler = {}
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        mask = self.label_data['case_id'].isin(split.tolist())
        df_metadata_slide = args.dataset_factory.label_data.loc[mask, :].reset_index(drop=True)

        omics_data_for_split = {}
        for key in args.dataset_factory.all_modalities.keys():
            
            raw_data_df = args.dataset_factory.all_modalities[key]
            mask = raw_data_df.index.isin(split.tolist())
            filtered_df = raw_data_df[mask]
            filtered_df = filtered_df[~filtered_df.index.duplicated()]
            filtered_df["temp_index"] = filtered_df.index
            filtered_df.reset_index(inplace=True, drop=True)

            clinical_data_mask = self.clinical_data.case_id.isin(split.tolist())
            clinical_data_for_split = self.clinical_data[clinical_data_mask]
            clinical_data_for_split = clinical_data_for_split.set_index("case_id")
            clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")

            mask = [True if item in list(filtered_df["temp_index"]) else False for item in df_metadata_slide.case_id]
            df_metadata_slide = df_metadata_slide[mask]
            df_metadata_slide.reset_index(inplace=True, drop=True)

            mask = [True if item in list(filtered_df["temp_index"]) else False for item in clinical_data_for_split.index]
            clinical_data_for_split = clinical_data_for_split[mask]
            clinical_data_for_split = clinical_data_for_split[~clinical_data_for_split.index.duplicated(keep='first')]

            filtered_normed_df = None
            if split_key in ["val"]:

                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]

                flat_df = np.expand_dims(df_for_norm.values.flatten(), 1)

                scaler_for_data = scaler[key]

                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

            elif split_key == "train":

                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]

                flat_df = df_for_norm.values.flatten().reshape(-1, 1)

                scaler_for_data = self._get_scaler(flat_df)

                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

                scaler[key] = scaler_for_data
                
            omics_data_for_split[key] = filtered_normed_df

        if split_key == "train":
            sample=True
        elif split_key == "val":
            sample=False
            
        split_dataset = SurvivalDataset(
            split_key=split_key,
            fold=fold,
            study_name=args.study,
            modality=args.modality,
            patient_dict=args.dataset_factory.patient_dict,
            metadata=df_metadata_slide,
            omics_data_dict=omics_data_for_split,
            data_dir= 'datasets_csv/{}/wsi_em'.format(args.study),
            num_classes=self.num_classes,
            label_col = self.label_col,
            censorship_var = self.censorship_var,
            valid_cols = valid_cols,
            is_training=split_key=='train',
            clinical_data = clinical_data_for_split,
            num_patches = self.num_patches,
            omic_names = self.omic_names,
            sample=sample
            )

        if split_key == "train":
            return split_dataset, scaler
        else:
            return split_dataset
    
    def __len__(self):
        return len(self.label_data)
    

class SurvivalDataset(Dataset):

    def __init__(self,
        split_key,
        fold,
        study_name,
        modality,
        patient_dict,
        metadata, 
        omics_data_dict,
        data_dir, 
        num_classes,
        label_col="survival_months",
        censorship_var = "censorship",
        valid_cols=None,
        is_training=True,
        clinical_data=-1,
        num_patches=4096,
        omic_names=None,
        sample=True,
        ): 

        super(SurvivalDataset, self).__init__()

        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        self.omics_data_dict = omics_data_dict
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        self.omic_names = omic_names
        self.num_pathways = len(omic_names)
        self.sample = sample

        self.slide_cls_id_prep()
    
    def _get_valid_cols(self):
        return self.valid_cols

    def slide_cls_id_prep(self):

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0]

            
    def __getitem__(self, idx):

        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        
        if self.modality == "SurvTransformer":
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            
            omic_list = []
            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))

            return (patch_features, omic_list, label, event_time, c, clinical_data, mask)
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)

    def get_data_to_return(self, idx):

        case_id = self.metadata['case_id'][idx]
        label = torch.Tensor([self.metadata['disc_label'][idx]]) # disc
        event_time = torch.Tensor([self.metadata[self.label_col][idx]])
        c = torch.Tensor([self.metadata[self.censorship_var][idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)

        return label, event_time, c, slide_ids, clinical_data, case_id
    
    def _load_wsi_embs_from_path(self, data_dir, slide_ids):

        patch_features = []
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi_path)
            patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)

        if self.sample:
            max_patches = self.num_patches

            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]

            if n_samples == max_patches:
                mask = torch.zeros([max_patches])
            else:
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.concat([patch_features, zeros], dim=0)
                mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
        
        else:
            mask = torch.ones([1])

        return patch_features, mask

    def get_clinical_data(self, case_id):

        try:
            age = self.clinical_data.loc[case_id, "age"]
        except:
            age = "N/A"
        
        try:
            site = self.clinical_data.loc[case_id, "site"]
        except:
            site = "N/A"

        try:
            is_female = self.clinical_data.loc[case_id, "is_female"]
        except:
            is_female = "N/A"
        
        clinical_data = (age, site, is_female)
        return clinical_data
    
    def getlabel(self, idx):

        label = self.metadata['label'][idx]
        return label

    def __len__(self):
        return len(self.metadata) 