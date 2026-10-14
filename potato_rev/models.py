import types
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from potato_rev import config as C

# ImageNet-1k V1 weights for every backbone, so all of them start from the same training recipe.
_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
}

# Hidden non-linearities found in the torchvision backbones. Sigmoid gates inside
# squeeze-and-excitation blocks are deliberately absent: they scale channels, they are not
# the layer non-linearity, and replacing them would break the [0, 1] gating.
_HIDDEN_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.SiLU)
_ALL_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.LeakyReLU, nn.PReLU, nn.Mish, nn.GELU)

_CLASS_OF = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "lrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "mish": nn.Mish,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
}


def make_activation(name: str) -> nn.Module:
    """One fresh module per call, so a learnable PReLU slope is never shared between sites."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "lrelu":
        return nn.LeakyReLU(negative_slope=C.LRELU_SLOPE, inplace=True)
    if name == "prelu":
        return nn.PReLU(num_parameters=1, init=0.25)
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "swish":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'. Supported: {sorted(_CLASS_OF)}")


def count_activations(model: nn.Module) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in model.modules():
        if isinstance(m, _ALL_ACTIVATIONS):
            counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    return counts


def replace_activations(module: nn.Module, name: str) -> int:
    """Recursively swap every hidden non-linearity module for `name`; returns the number swapped."""
    replaced = 0
    for child_name, child in module.named_children():
        if isinstance(child, _HIDDEN_ACTIVATIONS):
            setattr(module, child_name, make_activation(name))
            replaced += 1
        else:
            replaced += replace_activations(child, name)
    return replaced


def _densenet_forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.features(x)
    out = self.final_act(features)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return self.classifier(out)


def build_model(arch: str, activation: str, pretrained: bool = True,
                num_classes: int = C.NUM_CLASSES) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build a backbone, set its hidden non-linearity to `activation` at every site, and attach a
    fresh `num_classes`-way linear head. Returns the model and an audit of the replacement.
    """
    ctor, weights = _REGISTRY[arch]
    model = ctor(weights=weights if pretrained else None)
    native = C.NATIVE_ACTIVATION[arch]
    before = count_activations(model)

    if arch == "densenet121":
        # torchvision applies the last DenseNet non-linearity as a functional F.relu in forward(),
        # which module replacement cannot reach. Expose it as a module so it is swapped too.
        model.final_act = nn.ReLU(inplace=True)
        model.forward = types.MethodType(_densenet_forward, model)
        before["ReLU"] = before.get("ReLU", 0) + 1

    replaced = replace_activations(model, activation) if activation != native else 0

    if arch in ("resnet18", "resnet50"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch in ("mobilenet_v2", "efficientnet_b0", "efficientnet_b3"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:  # vgg16 / vgg19
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    after = count_activations(model)
    target = _CLASS_OF[activation].__name__
    audit = {
        "native_activation": native,
        "is_native": activation == native,
        "sites_before": before,
        "sites_after": after,
        "sites_replaced": replaced,
        # every hidden site must now be the requested type, and none may have been lost
        "verified": set(after) == {target} and after[target] == sum(before.values()),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
    }
    return model, audit
