import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

i = 0

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, patient_dir: Path, process_fun=None, id=None, skip_condition=None, cache=True):
        """
        Dataset class to store one patient's slices saved using np.savez_compressed.

        :param patient_dir Path to a dir containing saved slice .npz files.
        :param process_fun Processing on the fly function that takes the saved items as parameters
        :param id Patient id if available. For convenience and keeping track of patients.
        :param skip_condition predicate function to determine which slices should be excluded from the dataset.
         Takes the item after processing as an input.
        :param cache Whether to save/use cached lists of filtered indices to save time and avoid having to filter slices
         on object init.
        """

        self.patient_dir = patient_dir
        self.slice_paths = sorted(list(patient_dir.iterdir()))
        self.process = process_fun
        self.skip_condition = skip_condition
        self.id = id
        self.len = len(self.slice_paths)
        self.idx_map = {x: x for x in range(self.len)}

        if self.skip_condition is not None: # Perform some filtering based on 'skip_condition' predicate.
            import hashlib
            # try to generate a name for caching that depends on the patient_dir and skip_condition so that old cache is
            # not used when one of those is changed.
            # hashing of a function is tricky so this shouldn't be relied on too much...
            hash_object = hashlib.sha256((str(patient_dir)).encode("utf-8") + str(skip_condition.__code__.co_code).encode("utf-8"))
            name = hash_object.hexdigest()
            if (self.patient_dir.parent.parent / "valid_indices_cache" / f"{name}.pkl").exists() and cache:
                import pickle
                self.idx_map = pickle.load(open((self.patient_dir.parent.parent / "valid_indices_cache" / f"{name}.pkl"), "rb"))
                self.len = len(self.idx_map)
            else:
                # Try and find which slices should be skipped and thus determine the length of the dataset.
                valid_indices = []
                for idx in range(self.len):
                    print(idx)
                    global i
                    i = i + 1
                    with np.load(self.slice_paths[idx]) as data:
                        if self.process is not None:
                            item = self.process(**data)
                        else:
                            item = data
                        if not skip_condition(item):
                            valid_indices.append(idx)
                self.len = len(valid_indices)
                self.idx_map = {x: valid_indices[x] for x in range(self.len)}
                if cache:
                    import pickle
                    cache_dir = self.patient_dir.parent.parent / "valid_indices_cache"
                    cache_dir.mkdir(exist_ok=True)
                    pickle.dump(self.idx_map, open(cache_dir / f"{name}.pkl", "wb"))

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        path = str(self.slice_paths[idx])
        data = np.load(path)
        patient_str = "patient_"
        slice_str = "slice_"
        patient_id = int(path[path.find(patient_str)+len(patient_str):path.rfind("/")])
        slice_id = int(path[path.find(slice_str)+len(slice_str):path.rfind(".npz")])
        
        if self.process is not None:
            item = self.process(**data)
        else:
            item = (data,)
        return item + (patient_id, slice_id) 

    def __len__(self):
        return self.len


class BrainDataset(torch.utils.data.Dataset):

    def __init__(self, datapath: Path, dataset="brats2021_64x64", split="val", 
                 n_tumour_patients=None, n_healthy_patients=None,
                 scale_factor=1, binary=True, pad=None,
                 skip_healthy_s_in_tumour=False,
                 skip_tumour_s_in_healthy=True,
                 seed=0, cache=True, use_channels: List[int] = None, sequence_translation : bool = False,
                 norm_around_zero: bool = True):
        """
        Dataset class for training/evaluation with the option to have datasets for semi-supervision.

        :param dataset: dataset identifier
        :param split: "train", "val" or "test".
        :param n_tumour_patients: number of patients w/ tumours to use. All slices (including slices containing tumours)
         from these patients will be included in the dataset
        :param n_healthy_patients: number of patients w/o tumours to use. Only slices not containing any tumour gt will
         be included from these patients.
        :param scale_factor: For resizing data on the fly.
        :param binary: Whether to provide binary ground truth (background vs whole tumour) or possibly more granular
         classes (e.g. available in BraTS datasets).
        :param pad: Padding for data/gt.
        :param skip_healthy_s_in_tumour: Whether to skip healthy slices in "tumour" patients
         (e.g. for testing/visualising results).
        :param skip_tumour_s_in_healthy: whether to skip tumour slices in healthy patients. Usually yes, unless for debugging, etc.
        :param seed:
        :param cache: Whether to use caching for filtering slices
        :param use_channels: Whether to use a subset of modalities. E.g. [0, 1] for FLAIR and T1 in BraTS.
        """

        self.rng = random.Random(seed)
        self.sequence_translation = sequence_translation
        self.use_channels = use_channels if use_channels is not None else list(range(4))
        self.split = split
        train_path = datapath / "npy_train"
        val_path = datapath / "npy_val"
        test_path = datapath / "npy_test"

        if split == "train":
            path = train_path
        elif split == "val":
            path = val_path
        elif split == "test":
            path = test_path
        else:
            raise ValueError(f"split {split} unknown")


        # Assuming item[1] is the gt
        if binary:
            self.skip_tumour = lambda item: item[1].sum() > 0
            self.skip_healthy = lambda item: item[1].sum() < 1
        else:
            self.skip_tumour = lambda item: item[1][1:, ...].sum() > 5
            self.skip_healthy = lambda item: item[1][1:, ...].sum() < 5

        # On the fly preprocessing function
        def process(x, y, coords=None):
            if binary:
                # treat all tumour classes (or just WM lesions) as one for anomaly detection purposes.
                y = y > 0.5
            else:
                # convert to one-hot gt encoding
                y = np.concatenate([y == x for x in range(4)], axis=1)

            if scale_factor != 1: # Rescale on the fly if needed
                x = F.interpolate(torch.from_numpy(x).float(), scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)
                y = F.interpolate(torch.from_numpy(y).float(), scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)
                if coords is not None:
                    coords = F.interpolate(torch.from_numpy(coords), scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

            if pad is not None:
                x = F.pad(x, pad=pad)
                y = F.pad(y, pad=pad)
                if coords is not None:
                    coords = F.pad(coords, pad=pad)

            if norm_around_zero:
                x = x * 2 - 1

            # Convert to torch tensors of appropriate shape for the default collate in default pytorch dataloader.
            if coords is not None:
                return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float(), torch.from_numpy(coords[0]).float()
            else:
                return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float()

        patient_dirs = sorted(list(path.iterdir()))
        if split == "train":
            self.rng.shuffle(patient_dirs)

        #assert ((n_tumour_patients is not None) or (n_healthy_patients is not None))
        self.n_tumour_patients = n_tumour_patients if n_tumour_patients is not None else len(patient_dirs)
        self.n_healthy_patients = n_healthy_patients if n_healthy_patients is not None else len(patient_dirs) - self.n_tumour_patients

        # Patients with tumours (e.g. for a semi-supervised case where some tumour labels are provided for training)
        self.patient_datasets = [PatientDataset(patient_dirs[i], process_fun=process, id=i, cache=cache,
                                                  skip_condition=self.skip_healthy if skip_healthy_s_in_tumour else None)
                                 for i in range(self.n_tumour_patients)]

        # + only healthy slices from "healthy" patients
        self.patient_datasets += [PatientDataset(patient_dirs[i],
                                                 skip_condition=self.skip_tumour if skip_tumour_s_in_healthy else None,
                                                 cache=cache, process_fun=process, id=i)
                                  for i in range(self.n_tumour_patients, self.n_tumour_patients + self.n_healthy_patients)]

        self.dataset = ConcatDataset(self.patient_datasets)

    def __getitem__(self, idx):
        item = {}
        item['image'], item['gt'], item['coords'], item["patient_id"], item["slice_id"] = self.dataset[idx]
        
        # 'coords' is a Batch x 3 x Height x Width tensor providing a float in range (-1, 1) for a rough "location" of the
        # pixel in the 3D brain in the 3 axes. Can be ignored.
        if self.sequence_translation and (self.split != "test"):
           
            sequence = self.rng.choice(self.use_channels)
            item['image'] = item['image'][sequence].unsqueeze(0)
            item['y'] = sequence
        else:
            # if at least one voxel belongs has a tumour mask, it's an unhealthy slice
            is_slice_healthy = torch.amax(item['gt'],dim = (0,1,2)).to(torch.long)
            item['y'] = is_slice_healthy
            
        conditioning_x = item["image"].detach().clone()
        # passing brain mask for maintaining brain shape
        brain_mask = conditioning_x > -1

        if np.random.uniform() > 0.5:
            conditioning_x[...,:conditioning_x.shape[-2]//2,:] = -1
        else:
            conditioning_x[...,conditioning_x.shape[-2]//2:,:] = -1
        
        #torch.cat([conditioning_x,brain_mask[:1]],axis = 0)
        item["conditioning_x"] = brain_mask[:1].to(torch.float)
        
        return item

    def __len__(self):
        return len(self.dataset)

def re_write_file_names_to_include_healthy_status(path):
    paths = list(Path(path).glob("*/**.npz"))
