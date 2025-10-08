"""
ComfyUI节点实现
定义Nano Banana图像生成节点
"""

import torch
import random
import os
import tempfile
import logging
from typing import Any, Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .upload import upload_file_zh
    from .api_client import GrsaiAPI, GrsaiAPIError
    from .config import default_config
    from .utils import (
        download_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
    )
except ImportError:
    from upload import upload_file_zh
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import download_image, pil_to_tensor, format_error_message, tensor_to_pil


class SuppressFalLogs:
    """临时抑制FAL相关的详细HTTP日志的上下文管理器"""

    def __init__(self):
        self.loggers_to_suppress = [
            "httpx",
            "httpcore",
            "urllib3.connectionpool",
        ]
        self.original_levels = {}

    def __enter__(self):
        # 保存原始日志级别并设置为WARNING以上
        for logger_name in self.loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始日志级别
        for logger_name, original_level in self.original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)


# Size选择器节点
class NanoBanana_AspectRatioSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    default_config.SUPPORTED_ASPECT_RATIOS,
                    {"default": "1:1"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)
    FUNCTION = "execute"
    CATEGORY = "NuoAnAI/Nano Banana"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def execute(self, aspect_ratio):
        return (aspect_ratio,)


class NanoBanana_NumImagesSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_images": (
                    [1, 2, 3, 4],
                    {"default": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_images",)
    FUNCTION = "execute"
    CATEGORY = "NuoAnAI/Nano Banana"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def execute(self, num_images):
        return (num_images,)


class _GrsaiNanoBananaNodeBase:
    """
    所有Nano Banana节点的内部基类，处理通用逻辑。
    """

    FUNCTION = "execute"
    CATEGORY = "NuoAnAI/Nano Banana"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def _create_error_result(
        self, error_message: str, original_image: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        print(f"节点执行错误: {error_message}")
        if original_image is not None:
            image_out = original_image
        else:
            image_out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        return {
            "ui": {"string": [error_message]},
            "result": (image_out, f"失败: {error_message}"),
        }

    def _execute_generation(
        self,
        grsai_api_key: str,
        final_prompt: str,
        num_images: int,
        model: str,
        urls: list[str] = [],
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str]]:
        results_pil, result_urls, errors = [], [], []

        def generate_single_image(current_seed):
            try:
                api_client = GrsaiAPI(api_key=grsai_api_key)
                api_params = {
                    "prompt": final_prompt,
                    "model": model,
                    "urls": urls,
                }
                api_params.update(kwargs)
                pil_image, url = api_client.nano_banana_generate_image(**api_params)
                return pil_image, url
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
            # 限制seed在32位整数范围内，避免API解析错误
            seeds = [random.randint(1, 2147483647) for i in range(num_images)]
            future_to_seed = {
                executor.submit(generate_single_image, s): s for s in seeds
            }

            for future in as_completed(future_to_seed):
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        # 简化错误信息，不显示技术细节
                        errors.append(f"图像生成失败")
                    else:
                        pil_img, url = result
                        results_pil.append(pil_img)
                        result_urls.append(url)
                except Exception as exc:
                    errors.append(f"图像生成异常")

        return results_pil, result_urls, errors


# 节点1: 图片生成
class GrsaiNanoBanana_Generate(_GrsaiNanoBananaNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The character is sitting cross-legged on the sofa, and the Dalmatian is lying on the blanket sleeping.",
                    },
                ),
                "model": (
                    ["nano-banana-fast", "nano-banana"],
                    {"default": "nano-banana-fast"},
                ),
                "aspect_ratio": (
                    "STRING",
                    {"default": "auto"},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    def execute(self, **kwargs):
        images_in = [
            kwargs.get(f"image_{i}")
            for i in range(1, 6)
            if kwargs.get(f"image_{i}") is not None
        ]

        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(default_config.api_key_error_message)

        os.environ["NUOANAI_KEY"] = grsai_api_key

        uploaded_urls = []
        temp_files = []
        try:
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None:
                    continue

                pil_images = tensor_to_pil(image_tensor)
                if not pil_images:
                    continue

                with tempfile.NamedTemporaryFile(
                    suffix=f"_{i}.png", delete=False
                ) as temp_file:
                    pil_images[0].save(temp_file, "PNG")
                    temp_files.append(temp_file.name)

                with SuppressFalLogs():
                    uploaded_urls.append(upload_file_zh(temp_files[-1]))

            final_prompt = kwargs["prompt"]

        except Exception as e:
            return self._create_error_result(
                f"Image upload failed: {format_error_message(e)}"
            )
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

        num_images = kwargs.pop("num_images")
        model = kwargs.pop("model")
        kwargs.pop("prompt")
        for i in range(1, 6):
            kwargs.pop(f"image_{i}", None)

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key,
            final_prompt,
            num_images,
            model,
            uploaded_urls,
            **kwargs,
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}"
            )

        success_count = len(results_pil)
        final_status = f"生图 | 成功生成: {success_count}/{num_images} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


NODE_CLASS_MAPPINGS = {
    "NuoAnAINanoBanana_Generate": GrsaiNanoBanana_Generate,
    "NuoAnAINanoBanana_AspectRatioSelector": NanoBanana_AspectRatioSelector,
    "NuoAnAINanoBanana_NumImagesSelector": NanoBanana_NumImagesSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NuoAnAINanoBanana_Generate": "🎨 NuoAnAI Nano Banana - Generate",
    "NuoAnAINanoBanana_AspectRatioSelector": "📐 NuoAnAI Nano Banana - Aspect Ratio Selector",
    "NuoAnAINanoBanana_NumImagesSelector": "🔢 NuoAnAI Nano Banana - Num Images Selector",
}
