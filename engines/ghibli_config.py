"""
Ghibli Diffusion — 현재 서버(main.py)와 동일한 기본 설정.
다른 프로젝트로 엔진만 옮길 때 이 값으로 생성 후 stylize(image_bytes)만 호출하면
main.py POST /stylize 기본 폼값(controlnet_scale=1.0, steps=6, seed=42 등)과
동일한 인자로 pipeline이 호출됨. (검증: main.py 128~144행 get_engine + stylize 기본값)

사용 예:
    from engines.ghibli_diffusion_engine import GhibliDiffusionEngine
    from engines.ghibli_config import GHIBLI_DEFAULT_KWARGS

    engine = GhibliDiffusionEngine(**GHIBLI_DEFAULT_KWARGS)
    engine.load_models()
    result_bytes = engine.stylize(image_bytes)   # 추가 인자 없이 → 서버와 100% 동일
"""

# main.py /stylize 기본 요청 시 get_engine(…) + stylize(…) 에 전달되는 값과 동일
# (main은 strength/canny/ip_scale 미전달 → 엔진 기본값 사용 → 아래와 동일)
# 그 외 영향 있는 값 없음: prompt/negative_prompt·모델 ID·JPEG quality·전처리·device(cuda/자동) 는 엔진 코드에 고정.
GHIBLI_DEFAULT_KWARGS = {
    "content_size": 512,
    "controlnet_scale": 1.0,
    "guidance_scale": 2.0,       # Form 1.0 → 엔진 내부에서 2.0으로 올림
    "num_inference_steps": 6,
    "strength": 0.7,
    "canny_low": 100,
    "canny_high": 200,
    "ip_scale": 0.6,
    "default_seed": 42,
}
