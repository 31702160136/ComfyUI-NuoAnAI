"""
ComfyUI-NuoAnAI 节点注册入口
"""

from .flux_nodes import (
    NODE_CLASS_MAPPINGS as FLUX_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FLUX_NODE_DISPLAY_NAME_MAPPINGS,
)

from .gpt_image_nodes import (
    NODE_CLASS_MAPPINGS as GPT_IMAGE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GPT_IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
)

# 合并两组节点映射，供ComfyUI识别
NODE_CLASS_MAPPINGS = {
    **FLUX_NODE_CLASS_MAPPINGS,
    **GPT_IMAGE_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FLUX_NODE_DISPLAY_NAME_MAPPINGS,
    **GPT_IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
}

# 导出节点映射，供ComfyUI识别
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# 版本信息 - 必须与 pyproject.toml 中的版本保持完全一致
__version__ = "1.0.3"

# 插件作者信息
__author__ = "ComfyUI-NuoAnAI Team"
__description__ = "NuoAnAI图像生成节点 - 支持GPT Image，Flux.1 Kontext"
