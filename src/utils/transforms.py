import torch
from torchvision.transforms import Compose
from .data_process import world2cam, cam2world, project_2dto3d
from copy import deepcopy

class WorldToCameraCoord():
    def __call__(self, d):
        cam = d['camera'][d['meta']['cam_idx']]
        d["joint"] = world2cam(d["joint"], cam['R'], cam['t'])
        return d

class CameraToWorldCoord():
    def __init__(self):
        pass
    def __call__(self, d):
        cam = d['camera'][d['meta']['cam_idx']]
        d["joint"] = cam2world(d["joint"], cam['R'], cam['t'])
        return d

class CenterAroundJoint():
    def __call__(self, d):
        d["joint"] -= d["joint"][0]
        return d

class Create2DProjection():
    def __call__(self, d):
        assert "joint_2d" not in d
        cam = d['camera'][d['meta']['cam_idx']]
        d["joint_2d"] = project_2dto3d(d["joint"], cam['R'], cam['t'],
                                        cam['f'], cam['c'])
        return d

class ToTensor():
    def __init__(self, key=None):
        self.key = key
    def __call__(self, d):
        if self.key is None:
            return torch.tensor(d)
        else:
            d[self.key] = torch.tensor(d[self.key], dtype=torch.float32)
            return d

class RemoveKey():
    def __init__(self, key):
        self.key = key
    def __call__(self, d):
        d.pop(self.key)
        return d


class Normalize():
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), key=None):
        """
        Normalize input image with predefined mean/std.

        Parameters
        ----------
        mean: list, len=3
            mean values of (r, g, b) channels to use for normalizing.
        std: list, len=3
            stddev values of (r, g, b) channels to use for normalizing.
        """
        self.key = key
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, d):
        if self.key is None:
            return self.transform(d)
        else:
            d[self.key] = self.transform(d[self.key])
            return d

    def transform(self, d):
        for t, m, s in zip(d, self.mean, self.std):
            t.sub_(m).div_(s)
        return d

class UnNormalize():
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), key=None):
        """
        Normalize input image with predefined mean/std.

        Parameters
        ----------
        mean: list, len=3
            mean values of (r, g, b) channels to use for normalizing.
        std: list, len=3
            stddev values of (r, g, b) channels to use for normalizing.
        """
        self.key = key
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, d):
        if self.key is None:
            return self.transform(d)
        else:
            d[self.key] = self.transform(d[self.key])
            return d

    def transform(self, d):
        for t, m, s in zip(d, self.mean, self.std):
            t.mul_(s).add_(m)
        return d

normalizer = Compose([
    ToTensor(key="joint"),
    Normalize(
        mean=[0.0, 55.13095003627039, 212.47816798349012, 386.3335772451566, -55.13104568977698, 126.72167729445096, 304.8913198124597, -78.97671802658489, -181.2801294815762, -207.78378340281157, -246.60430851577476, -209.75715339168988, -161.0527482669428, -143.8003857718934, -99.31092875147561, 21.56259357082905, 90.71726208059506, ],
        std=[0.0001, 53.40743652057742, 149.1366647979204, 332.75411902514793, 53.407532660904494, 230.254330762241, 421.25536461584454, 109.23906078511108, 214.8082623854661, 261.86190967810535, 303.31403104452096, 136.2143440821917, 33.88532015723774, 168.9393335052613, 231.99935074785193, 193.72307460149764, 229.5416921918517, ],
        key="joint"
    ),
    ToTensor(key="joint_2d"),
    Normalize(
        mean=[82.22755572571894, 93.9711561937361, 129.87341934626798, 171.98573957744327, 70.08459998678214, 113.85067264434056, 156.82414275711153, 63.66131826686143, 38.2691350442841, 30.553163253327646, 21.40525341447111, 31.847708570472744, 44.23259513026189, 46.89965135501889, 57.49767904869024, 84.28654230153474, 97.62470687611409, ],
        std=[236.30116094600965, 223.3494911743147, 271.62922370952975, 317.3301851531254, 250.3553139763339, 294.1089524284145, 340.7590220640192, 209.40025179491502, 182.77950286744672, 171.35174577389512, 160.26081457946407, 205.08503533115072, 243.43688715065446, 260.4658866873485, 177.60084439532363, 194.28902158608352, 209.96135559785336, ],
        key="joint_2d"
    ),
])

unnormalizer = Compose([
    UnNormalize(
        mean=[0.0, 55.13095003627039, 212.47816798349012, 386.3335772451566, -55.13104568977698, 126.72167729445096, 304.8913198124597, -78.97671802658489, -181.2801294815762, -207.78378340281157, -246.60430851577476, -209.75715339168988, -161.0527482669428, -143.8003857718934, -99.31092875147561, 21.56259357082905, 90.71726208059506, ],
        std=[0.0001, 53.40743652057742, 149.1366647979204, 332.75411902514793, 53.407532660904494, 230.254330762241, 421.25536461584454, 109.23906078511108, 214.8082623854661, 261.86190967810535, 303.31403104452096, 136.2143440821917, 33.88532015723774, 168.9393335052613, 231.99935074785193, 193.72307460149764, 229.5416921918517, ],
        key="joint"
    ),
    UnNormalize(
        mean=[82.22755572571894, 93.9711561937361, 129.87341934626798, 171.98573957744327, 70.08459998678214, 113.85067264434056, 156.82414275711153, 63.66131826686143, 38.2691350442841, 30.553163253327646, 21.40525341447111, 31.847708570472744, 44.23259513026189, 46.89965135501889, 57.49767904869024, 84.28654230153474, 97.62470687611409, ],
        std=[236.30116094600965, 223.3494911743147, 271.62922370952975, 317.3301851531254, 250.3553139763339, 294.1089524284145, 340.7590220640192, 209.40025179491502, 182.77950286744672, 171.35174577389512, 160.26081457946407, 205.08503533115072, 243.43688715065446, 260.4658866873485, 177.60084439532363, 194.28902158608352, 209.96135559785336, ],
        key="joint_2d"
    ),    
])

trn_transforms = Compose([
    Create2DProjection(),
    WorldToCameraCoord(),
    CenterAroundJoint(),
    RemoveKey(key="camera"),
    normalizer,
])
val_transforms = Compose([
    Create2DProjection(),
    WorldToCameraCoord(),
    CenterAroundJoint(),
    RemoveKey(key="camera"),
    normalizer,
])
