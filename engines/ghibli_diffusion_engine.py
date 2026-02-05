"""Ghibli Diffusion: Img2Img + Canny(구조) + Ghibli(화풍) + LCM + IP-Adapter + Seed. 배경/인물 유지·눈 묘사 강화."""

import io
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Suppress diffusers safety checker warning
warnings.filterwarnings("ignore", message=".*safety_checker.*")

from .base_engine import BaseStyleEngine

LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"
PRECOMPUTED_EMBEDS_PATH = Path(__file__).resolve().parent.parent / "train_data" / "ip_adapter_embeds.pt"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"


class GhibliDiffusionEngine(BaseStyleEngine):
    """
    1. Img2Img + strength: 원본(배경·인물) 유지 강도 조절
    2. Canny: 형태/포즈 고정
    3. Ghibli + LCM + IP-Adapter + Seed
    """

    def __init__(
        self,
        content_size: int = 512,
        controlnet_scale: float = 0.85,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 8,
        strength: float = 0.7,
        canny_low: int = 100,
        canny_high: int = 200,
        ip_scale: float = 0.6,
        default_seed: int = 42,
        device: str | None = None,
    ):
        super().__init__(device)
        self.content_size = content_size
        self.controlnet_scale = controlnet_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.ip_scale = ip_scale
        self.default_seed = default_seed

        self._pipe = None
        self._precomputed_embeds = None  # train_data 사전 인코딩 (첫 생성용)
        self._first_gen_embeds = None  # 첫 스타일 결과 인코딩 (이후 일관성용)
        self._first_content_embeds = None  # 첫 프레임 입력(실제 인물) 인코딩 (이후 블렌드용, 한 번만 인코딩)

        self._default_prompt = (
            "black hair, exagerated eyes, big eyes"
            "ghibli style , cute anime character, large eyes, illustrated character"
            "soft features, expressive eyes, kawaii, stylized portrait"
            "studio ghibli character"
        )
        self._default_negative_prompt = (
            "ugly, blurry, low quality, distorted, deformed, realistic"
            "bad eyes, deformed eyes, blurry eyes, missing eyes, asymmetric eyes, poorly drawn eyes"
            "same as photo, copy of photo, photorealistic, small eyes"
        )

    @property
    def name(self) -> str:
        return "Ghibli Diffusion (Img2Img + Canny + LCM + IP-Adapter)"

    @property
    def speed_rating(self) -> str:
        return "fast"

    @property
    def quality_rating(self) -> int:
        return 5

    def load_models(self) -> None:
        """Img2Img + Canny + Ghibli + LCM LoRA + IP-Adapter 로드."""
        try:
            from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
            from diffusers import LCMScheduler
        except ImportError:
            raise ImportError(
                "diffusers required: pip install diffusers transformers accelerate"
            )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self._pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "nitrosocke/Ghibli-Diffusion",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self._pipe = self._pipe.to(self.device)

        # LCM: 속도 가속
        self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
        self._pipe.load_lora_weights(LCM_LORA)

        # IP-Adapter: 색/분위기 주입
        self._pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder="models",
            weight_name=IP_ADAPTER_WEIGHT,
        )
        self._pipe.set_ip_adapter_scale(self.ip_scale)

        # IP-Adapter 사용 시 enable_attention_slicing() 호출 금지.
        # SlicedAttnProcessor가 IP-Adapter processor를 덮어쓰면 encoder_hidden_states가 tuple일 때 AttributeError 발생.
        if self.device == "cuda":
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        # 사전 인코딩된 train_data 임베딩 로드 (첫 생성 시 사용)
        if PRECOMPUTED_EMBEDS_PATH.exists():
            data = torch.load(PRECOMPUTED_EMBEDS_PATH, map_location=self.device, weights_only=False)
            self._precomputed_embeds = data["embeds"]

    def _to_device(self, x, device):
        """임베딩을 디바이스로 이동."""
        if isinstance(x, (list, tuple)):
            return [self._to_device(t, device) for t in x]
        return x.to(device) if hasattr(x, "to") else x

    def _blend_embeds(self, embeds_a, embeds_b, alpha: float):
        """embeds_a 비중 alpha, embeds_b 비중 (1-alpha). 리스트/텐서 둘 다 지원."""
        if isinstance(embeds_a, (list, tuple)):
            return [
                self._blend_embeds(embeds_a[i], embeds_b[i], alpha)
                for i in range(len(embeds_a))
            ]
        return embeds_a * alpha + embeds_b * (1.0 - alpha)

    def _blend_embeds_3(self, embeds_a, embeds_b, embeds_c, w_a: float, w_b: float, w_c: float):
        """3-way: w_a*a + w_b*b + w_c*c (w_a + w_b + w_c = 1)."""
        if isinstance(embeds_a, (list, tuple)):
            return [
                self._blend_embeds_3(embeds_a[i], embeds_b[i], embeds_c[i], w_a, w_b, w_c)
                for i in range(len(embeds_a))
            ]
        return embeds_a * w_a + embeds_b * w_b + embeds_c * w_c

    def extract_canny_edge(self, image: Image.Image) -> Image.Image:
        """Canny 엣지 (형태 고정용)."""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """해상도 정리 (8의 배수)."""
        w, h = image.size
        if max(w, h) > self.content_size:
            ratio = self.content_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
        else:
            new_w, new_h = w, h
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    def set_prompts(self, prompt: str, negative_prompt: str | None = None) -> None:
        self._default_prompt = prompt
        if negative_prompt is not None:
            self._default_negative_prompt = negative_prompt

    def set_ip_scale(self, scale: float) -> None:
        self.ip_scale = scale
        if self._pipe is not None:
            self._pipe.set_ip_adapter_scale(scale)

    def stylize(
        self,
        content_image: Image.Image | bytes,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        controlnet_scale: float | None = None,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
        strength: float | None = None,
        seed: int | None = None,
        **kwargs
    ) -> bytes:
        """
        Img2Img: image=원본, control_image=canny. strength 낮을수록 배경·인물 유지.
        Canny + IP-Adapter 항상 사용. IP-Adapter 옵션은 생성자/서버 고정.
        """
        if self._pipe is None:
            self.load_models()

        if isinstance(content_image, bytes):
            content_image = Image.open(io.BytesIO(content_image)).convert("RGB")

        content_image = self.preprocess_image(content_image)
        canny_image = self.extract_canny_edge(content_image)

        prompt = prompt or self._default_prompt
        negative_prompt = negative_prompt or self._default_negative_prompt

        # 첫 생성 여부
        use_embeds = self._first_gen_embeds if self._first_gen_embeds is not None else self._precomputed_embeds
        is_first_gen = self._first_gen_embeds is None

        controlnet_scale = controlnet_scale if controlnet_scale is not None else self.controlnet_scale
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        num_inference_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        strength = strength if strength is not None else self.strength
        seed = seed if seed is not None else self.default_seed

        if guidance_scale <= 1.0:
            guidance_scale = 2.0

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # IP-Adapter: 첫 생성 → precomputed (0.8), 이후 → 첫 결과 (0.5로 낮춰서 보라색 방지)
        pipe_kw = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": content_image,
            "control_image": canny_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_scale,
            "generator": generator,
        }
        if use_embeds is not None:
            embeds = self._to_device(use_embeds, self.device)
            # 두 번째 이후: 첫 프레임 입력(저장된 실제 인물) 20% + precomputed 50% + 첫 결과 30%
            if not is_first_gen and self._precomputed_embeds is not None and self._first_content_embeds is not None:
                first_content_dev = self._to_device(self._first_content_embeds, self.device)
                precomputed_dev = self._to_device(self._precomputed_embeds, self.device)
                embeds = self._blend_embeds_3(
                    first_content_dev, precomputed_dev, embeds,
                    w_a=0.2, w_b=0.5, w_c=0.3,
                )
            pipe_kw["ip_adapter_image_embeds"] = embeds
        else:
            pipe_kw["ip_adapter_image"] = content_image

        # 첫 생성: ip_scale 그대로, 이후: 0.6 (블렌드 썼으므로 약간 상향)
        effective_scale = self.ip_scale if is_first_gen else 0.6
        self._pipe.set_ip_adapter_scale(effective_scale)

        with torch.no_grad():
            result = self._pipe(**pipe_kw)
            output_image = result.images[0]

        # 첫 생성 후: 결과 이미지 + 입력(실제 인물) 이미지 임베딩 각각 캐시 (이후 블렌드용, 매 프레임 인코딩 방지)
        if is_first_gen:
            self._first_gen_embeds = self._pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=output_image,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            self._first_gen_embeds = self._to_device(self._first_gen_embeds, "cpu")
            self._first_content_embeds = self._pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=content_image,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            self._first_content_embeds = self._to_device(self._first_content_embeds, "cpu")

        buffer = io.BytesIO()
        output_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
