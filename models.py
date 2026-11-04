import copy

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class Separable(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                         nn.BatchNorm2d(channels), nn.ReLU(inplace=False),
                         nn.Conv2d(channels, channels, 1, bias=False),
                         nn.BatchNorm2d(channels), nn.ReLU(inplace=False))


class Student(nn.Module):
    def __init__(self, channels=64, boundary=False, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        training = self.encoder.training
        self.encoder.eval()
        stages = {}
        with torch.no_grad():
            x = torch.zeros(1, 3, 128, 128)
            for i, block in enumerate(self.encoder):
                x = block(x)
                if 128 % x.shape[-1] == 0:
                    stages[128 // x.shape[-1]] = (i, x.shape[1])
        self.encoder.train(training)
        if not all(s in stages for s in (4, 8, 16, 32)):
            raise RuntimeError('MobileNet feature pyramid does not have required strides')
        self.stage_indices = [stages[s][0] for s in (4, 8, 16, 32)]
        self.stage_channels = [stages[s][1] for s in (4, 8, 16, 32)]
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in self.stage_channels)
        self.refine = nn.ModuleList(Separable(channels) for _ in range(4))
        self.classifier = nn.Conv2d(channels, 3, 1)
        self.boundary_head = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, padding=1),
                                           nn.ReLU(), nn.Conv2d(channels // 2, 1, 1)) if boundary else None

    def forward(self, x):
        shape = x.shape[-2:]
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.stage_indices:
                features.append(x)
        fused = self.refine[3](self.projections[3](features[3]))
        for i in (2, 1, 0):
            fused = self.refine[i](self.projections[i](features[i]) +
                                   F.interpolate(fused, size=features[i].shape[-2:], mode='bilinear', align_corners=False))
        logits = F.interpolate(self.classifier(fused), size=shape, mode='bilinear', align_corners=False)
        result = {'logits': logits}
        if self.boundary_head is not None:
            result['boundary'] = F.interpolate(self.boundary_head(fused), size=shape, mode='bilinear', align_corners=False)
        return result

    def inference_model(self):
        model = copy.deepcopy(self).eval()
        model.boundary_head = None
        return LogitsOnly(model).eval()


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


class Teacher(nn.Module):
    def __init__(self, variant='b2', pretrained=False, source=None):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if variant not in ('b0', 'b2'):
            raise ValueError('Teacher variant must be b0 or b2')
        if pretrained:
            name = source or 'nvidia/segformer-{}-finetuned-ade-512-512'.format(variant)
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                name, num_labels=3, ignore_mismatched_sizes=True)
        else:
            kwargs = {'hidden_sizes': [32, 64, 160, 256], 'depths': [2, 2, 2, 2], 'decoder_hidden_size': 256} if variant == 'b0' else {
                'hidden_sizes': [64, 128, 320, 512], 'depths': [3, 4, 6, 3], 'decoder_hidden_size': 768}
            self.network = SegformerForSemanticSegmentation(SegformerConfig(num_labels=3, **kwargs))

    def forward(self, x):
        output = self.network(pixel_values=x).logits
        return {'logits': F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)}


def build_model(config, pretrained=None):
    pretrain = config['pretrained'] if pretrained is None else pretrained
    if config['kind'] == 'teacher':
        return Teacher(config['variant'], pretrain, config.get('pretrained_source'))
    return Student(config['decoder_channels'], config['boundary'], pretrain)


def boundary_targets(mask, radius=1):
    valid = mask != 255
    edge = torch.zeros_like(valid)
    horizontal = (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (mask[:, 1:, :] != mask[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    if radius:
        edge = F.max_pool2d(edge[:, None].float(), 2 * radius + 1, 1, radius)[:, 0] > 0
    invalid = F.max_pool2d((~valid)[:, None].float(), 2 * (radius + 1) + 1, 1, radius + 1)[:, 0] > 0
    safe = valid & ~invalid
    return edge.float(), safe


def segmentation_loss(logits, mask):
    logits = logits.float()
    valid = mask != 255
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, mask, ignore_index=255, reduction='sum') / valid.sum()
    probs = logits.softmax(1)
    terms = []
    for c in (1, 2):
        prediction = probs[:, c] * valid
        target = (mask == c).float()
        if target.sum() > 0:
            terms.append(1 - (2 * (prediction * target).sum() + 1e-6) /
                         (prediction.sum() + target.sum() + 1e-6))
        else:
            terms.append(prediction.sum() / valid.sum())
    return ce + torch.stack(terms).mean()


def boundary_loss(prediction, mask, radius):
    target, valid = boundary_targets(mask, radius)
    values = prediction[:, 0].float()
    if not valid.any():
        return values.sum() * 0
    truth, pred = target[valid], values[valid]
    positives = truth.sum()
    negatives = truth.numel() - positives
    if positives > 0 and negatives > 0:
        weights = torch.where(truth > 0, 0.5 / positives, 0.5 / negatives)
        return (F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * weights).sum()
    return F.binary_cross_entropy_with_logits(pred, truth)


def distillation_loss(student, teacher, mask, temperature=4.0):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    student = student.float()
    teacher = F.interpolate(teacher.detach().float(), size=student.shape[-2:], mode='bilinear', align_corners=False)
    valid = mask != 255
    if not valid.any():
        return student.sum() * 0
    kl = F.kl_div(F.log_softmax(student / temperature, dim=1),
                  F.softmax(teacher / temperature, dim=1), reduction='none').sum(1)
    return kl[valid].mean() * temperature ** 2
