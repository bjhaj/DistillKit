import torch.nn as nn
from torch.nn import ReLU, ReLU6
import torch
from torchvision.models.mobilenetv2 import InvertedResidual
from torch.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def patch_residual_add(module):
    for name, m in module.named_modules():
        if isinstance(m, InvertedResidual):
            m.skip_add = FloatFunctional()
            original_forward = m.forward

            def new_forward(self, x):
                if self.use_res_connect:
                    return self.skip_add.add(x, self.conv(x))
                else:
                    return self.conv(x)

            m.forward = new_forward.__get__(m, m.__class__)


def replace_relu6_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, ReLU6):
            setattr(module, name, ReLU(inplace=True))
        else:
            replace_relu6_with_relu(child)



class QuantizableStudent(nn.Module):
    def __init__(self, dropout=0.1, num_classes=10):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Base MobileNetV2
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        base.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        base.classifier = nn.Sequential(nn.Linear(base.last_channel, num_classes))

        self.features = base.features
        self.classifier = base.classifier
        self.dropout = nn.Dropout(p=dropout)  # if you want to reintroduce dropout, modify here
        replace_relu6_with_relu(self)
        patch_residual_add(self)


    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling like MobileNetV2
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for idx, m in enumerate(self.features):
            if isinstance(m, nn.Sequential):
                for inner_idx in range(len(m) - 2):
                    if (
                        isinstance(m[inner_idx], nn.Conv2d)
                        and isinstance(m[inner_idx + 1], nn.BatchNorm2d)
                        and isinstance(m[inner_idx + 2], nn.ReLU6)
                    ):
                        fuse_modules(m, [str(inner_idx), str(inner_idx + 1), str(inner_idx + 2)], inplace=True)
    def quantize_model(
        self,
        train_loader,
        state_dict_path: str,
        save_path: str = "quantized_student.pth",
        max_calibration_batches: int = 20,
    ):
        # Load pre-trained weights
        state_dict = torch.load(state_dict_path)
        self.load_state_dict(state_dict)

        # Fuse modules
        self.fuse_model()

        # Set quantization config
        self.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

        # Prepare for static quantization
        torch.ao.quantization.prepare(self, inplace=True)

        # Calibration
        with torch.no_grad():
            for i, (inputs, _) in enumerate(train_loader):
                inputs = inputs.contiguous(memory_format=torch.channels_last)
                self(inputs)
                if i >= max_calibration_batches:
                    break

        # Convert to quantized version
        torch.ao.quantization.convert(self, inplace=True)

        # Save quantized model weights
        torch.save(self.state_dict(), save_path)
