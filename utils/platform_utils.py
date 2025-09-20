"""Platform-specific utilities and optimizations."""

import platform
import subprocess
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_platform_info() -> Dict[str, str]:
    """Get detailed platform information."""
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # Mac-specific information
    if info["system"] == "Darwin":
        try:
            # Get chip information
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            info["cpu_brand"] = result.stdout.strip()

            # Check if Apple Silicon
            if "Apple" in info["cpu_brand"] or info["machine"] == "arm64":
                info["is_apple_silicon"] = True
                info["recommended_backend"] = "torch"
                info["recommended_device"] = "mps"
            else:
                info["is_apple_silicon"] = False
                info["recommended_backend"] = "numpy"
                info["recommended_device"] = "cpu"

        except Exception as e:
            logger.debug(f"Could not get Mac CPU info: {e}")
            info["is_apple_silicon"] = info["machine"] == "arm64"

    elif info["system"] == "Linux" or info["system"] == "Windows":
        info["is_apple_silicon"] = False
        # Check for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                info["recommended_backend"] = "torch"
                info["recommended_device"] = "cuda"
            else:
                info["recommended_backend"] = "numpy"
                info["recommended_device"] = "cpu"
        except ImportError:
            info["recommended_backend"] = "numpy"
            info["recommended_device"] = "cpu"

    return info


def get_optimal_settings() -> Tuple[str, str]:
    """Get optimal backend and device for current platform."""
    platform_info = get_platform_info()

    backend = platform_info.get("recommended_backend", "numpy")
    device = platform_info.get("recommended_device", "cpu")

    logger.info(f"Platform: {platform_info['system']} {platform_info['machine']}")
    if platform_info.get("is_apple_silicon"):
        logger.info("Apple Silicon detected - MPS acceleration available")

    logger.info(f"Recommended settings: backend={backend}, device={device}")

    return backend, device


def optimize_for_mac() -> Dict[str, any]:
    """Get Mac-specific optimization settings."""
    settings = {}
    platform_info = get_platform_info()

    if platform_info.get("system") != "Darwin":
        return settings

    if platform_info.get("is_apple_silicon"):
        # Apple Silicon optimizations
        settings.update({
            "backend": "torch",
            "device": "mps",
            "batch_size": 256,  # MPS optimal batch size
            "num_threads": 8,   # Performance cores
        })
        logger.info("Applied Apple Silicon optimizations")
    else:
        # Intel Mac optimizations
        settings.update({
            "backend": "numpy",
            "device": "cpu",
            "batch_size": 128,
            "num_threads": 4,
        })
        logger.info("Applied Intel Mac optimizations")

    return settings


def check_backend_availability() -> Dict[str, bool]:
    """Check which backends are available."""
    available = {
        "numpy": True,  # Always available
        "torch": False,
        "cuda": False,
        "mps": False,
    }

    try:
        import torch
        available["torch"] = True

        if torch.cuda.is_available():
            available["cuda"] = True

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available["mps"] = True

    except ImportError:
        pass

    return available


def print_system_info():
    """Print system and backend information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)

    platform_info = get_platform_info()
    for key, value in platform_info.items():
        if not key.startswith("recommended"):
            print(f"{key:20}: {value}")

    print("\n" + "="*60)
    print("BACKEND AVAILABILITY")
    print("="*60)

    backends = check_backend_availability()
    for backend, available in backends.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{backend:20}: {status}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    backend, device = get_optimal_settings()
    print(f"Optimal backend     : {backend}")
    print(f"Optimal device      : {device}")

    if platform_info.get("is_apple_silicon"):
        print("\nNote: Your Mac has Apple Silicon (M1/M2/M3/M4)")
        print("      MPS acceleration is recommended for best performance")

    print("="*60 + "\n")