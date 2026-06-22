""" Termal xavfsizlik moduli.
GPU (pynvml yoki nvidia-smi) va CPU (psutil) haroratini kuzatadi.
Harorat chegaradan oshsa trening AVTOMATIK PAUZA qilinadi va harorat
"resume" darajasiga tushguncha kutiladi. Google Colab T4 va lokal
mashinalarning ikkalasida ham ishlaydi; sensor topilmasa jimgina o'tadi. """

from __future__ import annotations
import time
import subprocess
from typing import Callable, Optional, Tuple

def _gpu_temp_pynvml() -> Optional[int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None

def _gpu_temp_smi() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None

def gpu_temperature() -> Optional[int]:
    t = _gpu_temp_pynvml()
    return t if t is not None else _gpu_temp_smi()

def cpu_temperature() -> Optional[int]:
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        vals = [e.current for entries in temps.values() for e in entries if e.current is not None]
        return int(max(vals)) if vals else None
    except Exception:
        return None

class ThermalGuard:
    def __init__(self, gpu_max: int = 79, gpu_resume: int = 68,
        cpu_max: int = 90, cpu_resume: int = 78,
        check_seconds: float = 20.0,
        log: Callable[[str], None] = print) -> None:
        self.gpu_max = gpu_max
        self.gpu_resume = gpu_resume
        self.cpu_max = cpu_max
        self.cpu_resume = cpu_resume
        self.check_seconds = check_seconds
        self.log = log
        self._last_check = 0.0

    def temps(self) -> Tuple[Optional[int], Optional[int]]:
        return gpu_temperature(), cpu_temperature()

    def _too_hot(self) -> bool:
        g, c = self.temps()
        return (g is not None and g >= self.gpu_max) or \
        (c is not None and c >= self.cpu_max)

    def _cooled(self) -> bool:
        g, c = self.temps()
        ok_g = g is None or g <= self.gpu_resume
        ok_c = c is None or c <= self.cpu_resume
        return ok_g and ok_c

    def wait_if_hot(self) -> float:
        if not self._too_hot():
            return 0.0
        g, c = self.temps()
        self.log(f"[XAVFSIZLIK] Harorat yuqori (GPU={g}C CPU={c}C) — trening PAUZA qilindi, sovushini kutyapmiz...")
        t0 = time.time()
        while not self._cooled():
            time.sleep(self.check_seconds)
            g, c = self.temps()
            self.log(f"[XAVFSIZLIK] kutilmoqda... GPU={g}C CPU={c}C")
        waited = time.time() - t0
        self.log(f"[XAVFSIZLIK] Harorat normallashdi, trening davom etadi "
        f"({waited:.0f}s kutildi).")
        return waited

    def periodic_check(self, min_interval: float = 30.0) -> float:
        now = time.time()
        if now - self._last_check < min_interval:
            return 0.0
        self._last_check = now
        return self.wait_if_hot()