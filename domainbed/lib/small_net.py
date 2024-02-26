import torch
import timm
import transformers
from transformers import ConvNextV2ForImageClassification
from domainbed.lib.vision_transformer import Identity
from transformers import AutoImageProcessor


class MobileNetV3(torch.nn.Module):
    KNOWN_MODELS = {
        'MobileNetV3': timm.models.mobilenetv3.mobilenetv3_small_100
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        self.n_outputs = 1000
        # self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    

class EfficientNetV2(torch.nn.Module):
    KNOWN_MODELS = {
        'EfficientNetV2': timm.models.efficientnet.efficientnet_b3
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm.normalized_shape[0]
        self.n_outputs = 1000
        self.network.head = Identity()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)


class ConvNext2(torch.nn.Module):
    KNOWN_MODELS = {
        'ConvNext2': transformers.ConvNextV2Model
    }

    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224",do_resize=False,  )#do_normalize=False, do_resize=False, do_crop=False, do_scale=False, do_flip=False,do_rotate=False,do_translate=False,do_resize_to=None,
        convnext2= ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
        for name, param in convnext2.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
        self.network = convnext2.to(device=device)
        self.network.train()

        # self.n_outputs = self.network.norm_pre.norm.normalized_shape[0]
        self.n_outputs = 1000
        self.hparams = hparams


    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.tokenizer(x, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = self.network(**x)
        x = x.logits
        return x


class ConvNext(torch.nn.Module):
    KNOWN_MODELS = {
        'ConvNext': timm.models.convnext.convnext_base_in22k
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm_pre.norm.normalized_shape[0]
        self.n_outputs = 21841
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
    
class RegNetY(torch.nn.Module):
    KNOWN_MODELS = {
        'RegNetY': timm.models.regnet.regnety_160,
    }
    def __init__(self, input_shape, hparams):
        super().__init__()
        func = self.KNOWN_MODELS[hparams['backbone']]
        self.network = func(pretrained=True)
        # self.n_outputs = self.network.norm_pre.norm.normalized_shape[0]
        self.network.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1),
        )
        self.n_outputs = 3024
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)