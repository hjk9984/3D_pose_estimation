import torch
import torch.nn as nn

class BaseBlock(nn.Module):
    def __init__(self, d_in, d_out, prob):
        super().__init__()

        self.layers = nn.Sequential(*self._make_blocks(d_in, d_out, prob))

        if d_in == d_out:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )

    def _make_blocks(self, d_in, d_out, prob):
        layers = []
        for _ in range(2):
            layers.extend([
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True),
                nn.Dropout(prob)
            ])
            d_in = d_out

        return layers

    def forward(self, x):
        res = self.residual(x)
        output = self.layers(x)
        return res + output


class Decoder(nn.Module):
    def __init__(self, n_joints, fc_ch, prob):
        super().__init__()

        self.start_fc = nn.Linear(n_joints*2, 1024)
        self.blocks = nn.Sequential(
            *[BaseBlock(fc_ch[i], fc_ch[i+1], prob) for i in range(len(fc_ch) - 1)]
            )
        self.end_fc = nn.Linear(1024, n_joints*3)
    
    def forward(self, x):
        x = self.start_fc(x)
        x = self.blocks(x)
        x = self.end_fc(x)
        return x

def build_estimator_baseline(cfg):
    # TODO: considering whether insert freeze or not
    n_joints = cfg.INPUT.N_JOINTS
    channels = cfg.MODEL.BASELINE_2DTO3D.CHANNELS
    prob = cfg.MODEL.DROPOUT_P

    return Decoder(n_joints, channels, prob)


if __name__ == "__main__":
    tmp2 = Decoder(17, [1024, 2048, 1024], 0.3)
    print(tmp2)

