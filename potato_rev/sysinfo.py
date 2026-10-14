import os
import platform
from typing import Any, Dict

import torch
import torchvision


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                return round(int(f.readline().split()[1]) / 2 ** 20, 1)
        except OSError:
            return float("nan")


def system_specs() -> Dict[str, Any]:
    """Hardware and software configuration, recorded alongside the results for reproducibility."""
    cuda = torch.cuda.is_available()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1) if cuda else None,
        "cpu": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
    }
