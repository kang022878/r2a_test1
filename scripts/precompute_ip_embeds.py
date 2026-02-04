#!/usr/bin/env python3
"""
train_data/ 내 이미지들을 IP-Adapter용으로 사전 인코딩하여 저장합니다.

실행: python scripts/precompute_ip_embeds.py

출력: train_data/ip_adapter_embeds.pt
- embeds: 10장 평균 임베딩 → ip_adapter_image_embeds로 바로 전달
- embeds_list: 이미지별 개별 임베딩 (선택 사용)

사용 예:
  data = torch.load("train_data/ip_adapter_embeds.pt", map_location="cuda", weights_only=False)
  pipe(..., ip_adapter_image_embeds=data["embeds"])
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image


def main():
    train_dir = ROOT / "train_data"
    out_path = train_dir / "ip_adapter_embeds.pt"

    # 이미지 경로 수집 (data1, data2, ... data10 순)
    image_paths = sorted(
        train_dir.glob("*.png"),
        key=lambda p: (p.stem.replace("data", "").zfill(4), p.name),
    )
    if not image_paths:
        print(f"No .png files in {train_dir}")
        sys.exit(1)

    print(f"Loading pipeline (same as Ghibli engine)...")
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )

    # 이미지별로 인코딩 (IP-Adapter 기본은 1개 이미지/어댑터)
    all_embeds = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        print(f"  Encoding {p.name}...")
        emb = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=img,
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        all_embeds.append(emb)

    # 평균 임베딩: 여러 이미지 스타일을 하나로 융합하여 바로 사용 가능
    # prepare 반환형: list of tensors (예: [tensor(2,1,1024)])
    if isinstance(all_embeds[0], (list, tuple)):
        merged = [
            torch.stack([e[i] for e in all_embeds]).mean(dim=0).to(all_embeds[0][i].dtype)
            for i in range(len(all_embeds[0]))
        ]
    else:
        merged = torch.stack(all_embeds).mean(dim=0)

    # 저장: 바로 ip_adapter_image_embeds로 사용 가능한 형태 (CPU로 저장)
    def to_cpu(x):
        if isinstance(x, (list, tuple)):
            return [to_cpu(t) for t in x]
        return x.cpu() if hasattr(x, "cpu") else x

    save_data = {
        "embeds": to_cpu(merged),  # 평균 → ip_adapter_image_embeds로 바로 전달
        "embeds_list": to_cpu(all_embeds),  # 개별 이미지별 임베딩 (선택 사용)
        "image_names": [p.name for p in image_paths],
    }
    torch.save(save_data, out_path)
    print(f"Saved to {out_path}")

    # 로드 테스트
    loaded = torch.load(out_path, map_location=device, weights_only=False)
    print(f"Verify: loaded embeds for {len(loaded['image_names'])} images")


if __name__ == "__main__":
    main()
