import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR


from src.modeling.model_factory import model_dict
from src.data.human36m import build_dataset, TRAIN_SUBJECTS, TEST_SUBJECTS
from src.utils.transforms import trn_transforms, val_transforms
from opt import base_parse
from src.config import cfg
from src.utils.metric import mpjpe

class Tester:
    def __init__(self, cfg, args):
        # self.dataset = NormHuman36mDataset(TEST_SUBJECTS, args.data_path, is_train=False)
        self.dataset = build_dataset(cfg, args, is_train=False)
        self.loader = DataLoader(self.dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH)
    
    def test(self, model, device):
        error = 0
        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                error += x.shape[0] * x.shape[1] * mpjpe(output, y).item()

        final_error = error / len(self.dataset) * 1000
        print(f"error : {final_error}")


class Trainer:
    '''
        Defining optimizer, dataloader etc. for the pipeline of training
    '''
    def __init__(self, cfg, args):
        self.model = model_dict[cfg.MODEL.HEAD](cfg)
        # self.dataset = NormHuman36mDataset(TRAIN_SUBJECTS, args.data_path)
        self.dataset = build_dataset(cfg, args, is_train=True)
        print(len(self.dataset))
        self.loader = DataLoader(self.dataset, batch_size=256, shuffle=True)
        self.epoch = cfg.SOLVER.MAX_EPOCH
        self.step = len(self.dataset) // cfg.SOLVER.IMS_PER_BATCH

        self.tester = Tester(cfg, args)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.SOLVER.BASE_LR
        )
        self.scheduler = ExponentialLR(self.optimizer, 0.9)
        self.loss_fn = nn.MSELoss()

        # for m1 chip
        self.device = "mps" if torch.backends.mps.is_available() else 'cpu'
        print(f"torch device : {self.device}")


    def train(self):
        self.model.to(self.device)
        for cur_epoch in range(self.epoch):
            for idx, data in enumerate(self.loader):
                x, y = data
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

                if idx % 100 == 0:
                    print(f"{cur_epoch} epoch [{idx}/{self.step}] Loss : {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.5f}")

            if cur_epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                torch.save(self.model.state_dict(), f"./checkpoints/{cfg.MODEL.HEAD}_{cur_epoch}.pth")
                print(f'model saved at ./checkpoints/{cfg.MODEL.HEAD}_{cur_epoch}.pth')

            if cur_epoch in cfg.SOLVER.STEPS:
                self.scheduler.step()
            
            if cur_epoch % cfg.TEST_PERIOD == 0:
                self.tester.test(self.model, self.device)


def main():
    args = base_parse()
    print(args)

    #tmp
    args.config = './configs/simple3d_baseline.yaml'

    cfg.merge_from_file(args.config)
    if not args.opts is None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    Trainer(cfg, args).train()

if __name__ == "__main__":
    main()
