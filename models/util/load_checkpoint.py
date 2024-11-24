import torch
from typing import Dict, Optional, List
from collections import OrderedDict

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
    strict: bool = False,
    param_mapping: Optional[List[str]] = None,
    verbose: bool = True
) -> None:
    """
    灵活加载 PyTorch 模型的 checkpoint。
    
    Args:
        model: 要加载参数的模型
        checkpoint_path: checkpoint 文件路径
        map_location: 加载设备
        strict: 是否严格匹配参数名称
        param_mapping: 参数名称映射字典, 用于重命名参数
        verbose: 是否打印详细信息
    """
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # 获取 state_dict, 处理不同的 checkpoint 格式
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # 创建新的 state_dict 用于加载
    new_state_dict = OrderedDict()
    
    # 处理参数名称映射
    if param_mapping is not None:
        for mp in param_mapping:
            old_key, new_key = mp.split("->")
            for key in state_dict:
                if old_key in key:
                    new_state_dict[key.replace(old_key, new_key)] = state_dict[key]
                elif key not in new_state_dict:
                    new_state_dict[key] = state_dict[key]
    else:
        for k, v in state_dict.items():
            new_state_dict[k] = v
    
    # 获取模型当前的参数名称
    model_state_keys = set(model.state_dict().keys())
    # 获取要加载的参数名称
    checkpoint_state_keys = set(new_state_dict.keys())
    
    if verbose:
        # 打印加载信息
        missing_keys = model_state_keys - checkpoint_state_keys
        unexpected_keys = checkpoint_state_keys - model_state_keys
        
        if len(missing_keys) > 0:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        print(f"Successfully loaded {len(new_state_dict)} parameters")
    
    # 加载参数
    try:
        model.load_state_dict(new_state_dict, strict=strict)
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise e
    
    return model