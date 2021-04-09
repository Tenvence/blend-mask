import torch

from model import BlendMask


def main():
    model = BlendMask(num_classes=80, num_channels=256, num_basis=4, attention_size=56)
    inp = torch.zeros((2, 3, 512, 512))
    class_pred, centerness_pred, distances_pred, attentions_pred = model(inp)
    print(class_pred.shape, centerness_pred.shape, distances_pred.shape, attentions_pred.shape)


if __name__ == '__main__':
    main()
