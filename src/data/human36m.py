import os
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import numpy as np

from tqdm import tqdm
from PIL import Image
import json
from ..utils.transforms import trn_transforms, val_transforms
# 싱글이미지만 가능하게
# 여러 프레임이 한번에 들어가게
# multi view


TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

ACTIONS = {"Directions":1,
            "Discussion":2,
            "Eating":3,
            "Greeting":4,
            "Phoning":5,
            "Photo":6,
            "Posing":7,
            "Purchases":8,
            "Sitting":9,
            "SittingDown":10,
            "Smoking":11,
            "Waiting":12,
            "WalkDog":13,
            "Walking":14,
            "WalkTogether":15}

# TODO: action and subject?

class Human36mDataset(Dataset):
    def __init__(self, subjects, data_path, is_train=True, 
                transforms=trn_transforms, target_transforms=lambda x:x):
        '''
            subjects: a list contained the numbers of subjects 
            data_path: root data path 
        '''
        # subjects = TRAIN_SUBJECTS if is_train else TEST_SUBJECTS
        print(f"dataset is consisted of the subjects {subjects}")
        self.data_path = os.path.join(data_path, "human36m")
        self.is_train = is_train
        self.cameras = {}
        self._data = []
        self.transforms = transforms
        self.target_transforms = target_transforms

        # annotation parsing
        for subject_id in subjects:
            sub_dict = self._get_sub_anno(subject_id)
            self.cameras[subject_id] = {idx: np.array(value) for idx, value in sub_dict['camera'].items()}
            
            cur_subaction = -1
            period = 0
            end_idx = 0
            for sample_idx, sample_meta in tqdm(enumerate(sub_dict['data']['images'])):
                action_idx = str(sample_meta['action_idx'])
                subaction_idx = str(sample_meta["subaction_idx"])
                frame_idx = str(sample_meta["frame_idx"])
                camera_idx = str(sample_meta["cam_idx"])

                if cur_subaction != subaction_idx:
                    period = len(sub_dict['joint'][action_idx][subaction_idx])
                    cur_subaction = subaction_idx
                    end_idx += 4 * period
                sample_meta['cam_period'] = period
                sample_meta['end_idx'] = end_idx

                self._data.append({
                    "joint": np.array(sub_dict['joint'][action_idx][subaction_idx][frame_idx]), # (17, 3)
                    "meta": sample_meta,
                    "bbox": sub_dict["data"]["annotations"][sample_idx],
                    "camera": self.cameras[subject_id][camera_idx],
                })


    def _get_sub_anno(self, sub):
        def _json_parse(path):
            with open(path, 'r') as f:
                json_obj = json.load(f)
            assert json_obj is not None, f"{path} is not existed"
            return json_obj

        base_path = os.path.join(self.data_path, "annotations", "Human36M_subject")
        return {
            "data": _json_parse(base_path + f"{sub}_data.json"),
            "camera": _json_parse(base_path + f"{sub}_camera.json"),
            "joint": _json_parse(base_path + f"{sub}_joint_3d.json")
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        target = self._data[index]
        twoD_input = self.transforms(target["joint"])

        return twoD_input, target

class ImgHuman36mDataset(Human36mDataset):
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, "images", self._data[index]['meta']['file_name'])).convert("RGB")
        # target = self._data[index]['joint']
        target = self._data[index]

        img = self.transforms(img)
        # target = self.target_transforms(target)

        return img, target

# sub-action
# cam-idx
# frame

class MultiviewTemporalSampler(object):
    def __init__(self, dataset, n_multi, n_tempor):
        self.anno = dataset._data
        self.n_data = len(dataset)
        self.n_multi = n_multi
        self.n_tempor = n_tempor

    def __len__(self):
        return self.n_data

    def __iter__(self):
        while True:
            idx = 0
            while True:
                idx = np.random.randint(self.n_data)
                if self.anno[idx]['meta']['end_idx'] >= idx + self.n_tempor:
                    break
        
            period = self.anno[idx]['meta']['cam_period']
            cam_idx = self.anno[idx]['meta']['cam_idx']
            indices = torch.tensor([
                idx + (i-cam_idx+1)*period + j
                for i in range(self.n_multi) for j in range(self.n_tempor)
            ])
            yield indices


class NormHuman36mDataset(Dataset):
    '''
        Normalized 2d coord input(not image), 3d coord output
    '''
    def __init__(self, actions, data_path, use_hg=True, is_train=True, 
                transforms=None, target_transforms=None):
        """
        :param actions: list of actions to use
        :param data_path: path to root dir of dataset
        :param use_hg: use stacked hourglass detections
        :param is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = os.path.join(data_path, "normHuman36m")

        self.is_train = is_train
        self.use_hg = use_hg

        self.inputs = []
        self.outputs = []

        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'
        
        if self.is_train:
            self._data_setting(train_2d_file, 'train_3d.pth.tar')
        else:
            self._data_setting(test_2d_file, 'test_3d.pth.tar')

    def _data_setting(self, path_2d, path_3d):
        data_3d = torch.load(os.path.join(self.data_path, path_3d))
        data_2d = torch.load(os.path.join(self.data_path, path_2d))
        for k2d in data_2d.keys():
            (sub, act, fname) = k2d           
            if ACTIONS[act] not in self.actions: # eager operation, not short circuit in python
                continue

            k3d = k2d
            k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
            num_f, _ = data_2d[k2d].shape
            assert data_2d[k2d].shape[0] == data_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
            for i in range(num_f):
                self.inputs.append(data_2d[k2d][i])
                self.outputs.append(data_3d[k3d][i])


    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs[index]).float()
        outputs = torch.from_numpy(self.outputs[index]).float()
        return inputs, outputs

    def __len__(self):
        return len(self.inputs)


DATASET_DICT = {
    'Human36m': Human36mDataset,
    'NormHuman36m': NormHuman36mDataset
}

def build_dataset(cfg, args, *, is_train):
    dataset_list = []

    if is_train:
        for dataset_name in cfg.DATASETS.TRAIN:
            dataset_list.append(DATASET_DICT[dataset_name](TRAIN_SUBJECTS, args.data_path))
    else:
        for dataset_name in cfg.DATASETS.TEST:
            dataset_list.append(DATASET_DICT[dataset_name](TEST_SUBJECTS, args.data_path))
    
    dataset = ConcatDataset(dataset_list)
    return dataset
    
    

if __name__ == "__main__":
    # import pickle
    path = '/Users/antae/Dev_hj/data/'

    import json
    import os


    dataset = Human36mDataset([1], path)
    sampler = MultiviewTemporalSampler(dataset, 4, 9)
    cnt = 0

    print(dataset[0][1]['joint'])
    # for indices in sampler:
    #     tmp = [dataset[i][1]["meta"] for i in indices]
    #     for data in tmp:
    #         print(data['id'], data["subaction_idx"], data["cam_idx"], data['frame_idx'])
    #     print()
    #     cnt += 1
    #     if cnt > 10:
    #         break

    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0][1]['joint'].shape)
    # cnt = 0
    # print(dataset[5531][1])
    # print(dataset[5532][1])
    # for _, y in dataset:
    #     meta = y['meta']
    #     if cnt % 100 == 0:
    #         print(meta['id'], meta['subaction_idx'], meta['cam_idx'], meta['frame_idx'], meta["cam_period"])
    #     if cnt > 10000:
    #         break
    #     cnt += 1



