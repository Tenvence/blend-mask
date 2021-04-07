import torch

from model.modules import StageBackbone, FeaturePyramidNet


def main():
    backbone = StageBackbone()
    fpn = FeaturePyramidNet(in_channels_list=[512, 1024, 2048], out_channels=256)
    inp = torch.zeros((2, 3, 512, 512))
    stage_features_dict = backbone(inp)
    fpn_out = fpn(stage_features_dict)
    for key, val in fpn_out.items():
        print(key, val.shape)


if __name__ == '__main__':
    main()
