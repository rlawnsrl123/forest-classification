"""환경 점검 — 5090(Blackwell, sm_120)에서 실제로 연산이 도는지까지 확인한다.

torch가 설치돼도 구버전 휠이면 sm_120 커널이 없어 import는 되는데 연산에서 죽는다.
그래서 버전만 찍지 말고 conv + backward를 한 번 돌려본다.
"""
import sys


def main():
    print("python", sys.version.split()[0])
    import numpy as np
    print("numpy ", np.__version__)

    import torch
    print("torch ", torch.__version__, "| cuda", torch.version.cuda)
    if not torch.cuda.is_available():
        print("!! CUDA 사용 불가 — CPU 휠이 설치됐을 수 있습니다")
        return 1

    i = torch.cuda.current_device()
    print("gpu   ", torch.cuda.get_device_name(i))
    cap = torch.cuda.get_device_capability(i)
    print("compute capability sm_%d%d" % cap)
    print("지원 아키텍처", torch.cuda.get_arch_list())
    if "sm_%d%d" % cap not in torch.cuda.get_arch_list():
        print("!! 이 휠에 sm_%d%d 커널이 없습니다 — CUDA 12.8+ 빌드를 설치하세요" % cap)
        return 1

    # 실제 연산 + bf16 autocast + backward
    x = torch.randn(8, 20, 128, 128, device="cuda", requires_grad=True)
    conv = torch.nn.Conv2d(20, 32, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = conv(x).square().mean()
    y.backward()
    torch.cuda.synchronize()
    print("conv+backward OK  | bf16 autocast OK")
    print("VRAM %.1f GB" % (torch.cuda.get_device_properties(i).total_memory / 1e9))

    for mod in ("rasterio", "sklearn", "segmentation_models_pytorch"):
        try:
            m = __import__(mod)
            print("%-28s %s" % (mod, getattr(m, "__version__", "ok")))
        except Exception as e:
            print("%-28s 없음 (%s)" % (mod, type(e).__name__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
