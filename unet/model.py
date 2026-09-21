"""모델 생성. 기본은 자체 U-Net(의존성 0), --arch smp:<encoder> 로 사전학습 백본.

실험 3(위성 사전학습 인코더 비교)에서 백본만 갈아끼우면 되도록 분리했다.
"""
import torch
import torch.nn as nn


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    """표준 U-Net. BatchNorm 필수 — AdaBN 실험이 이 층의 통계를 건드린다."""

    def __init__(self, in_ch=20, n_cls=3, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottom = _block(c, chs[-1])
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.convs.append(_block(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(c, n_cls, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, cv, s in zip(self.ups, self.convs, reversed(skips)):
            x = cv(torch.cat([up(x), s], 1))
        return self.head(x)


def build_model(arch="unet", in_ch=20, n_cls=3, base=32, depth=4):
    if arch == "unet":
        return UNet(in_ch, n_cls, base=base, depth=depth)
    if arch.startswith("smp:"):
        import segmentation_models_pytorch as smp
        enc = arch.split(":", 1)[1]
        weights = "imagenet" if enc.endswith("+pre") else None
        enc = enc.replace("+pre", "")
        return smp.Unet(encoder_name=enc, encoder_weights=weights,
                        in_channels=in_ch, classes=n_cls)
    raise ValueError("알 수 없는 arch: " + arch)


def reset_bn(model):
    """BatchNorm running 통계 초기화 — AdaBN 1단계. 되돌린 층 수를 반환."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None      # 누적 평균 → 본 만큼 전부 반영
            n += 1
    return n
