# resource_analysis.py
import time
import threading
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
import torch
import wandb

# Optional deps
try:
    import psutil
    _PSUTIL_OK = True
except Exception:
    psutil = None
    _PSUTIL_OK = False

try:
    import pynvml
    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False

# FLOPs via torch.profiler (best-effort)
try:
    from torch.profiler import profile as torch_profile
    from torch.profiler import ProfilerActivity
    _PROFILER_OK = True
except Exception:
    torch_profile = None
    ProfilerActivity = None
    _PROFILER_OK = False


def profiler_available() -> bool:
    return bool(_PROFILER_OK)


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def device_cuda_index(device: torch.device) -> Optional[int]:
    try:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            return int(str(device).split(":")[1])
    except Exception:
        pass
    return None


class ResourceMonitor:
    def __init__(
        self,
        cuda_index: Optional[int],
        log_every_s: float = 1.0,
        disk_path: str = ".",
        prefix: str = "sys",
        step_fn: Optional[Callable[[], Optional[int]]] = None,
        log_to_wandb: bool = True,
    ):
        self.cuda_index = cuda_index
        self.log_every_s = float(log_every_s)
        self.disk_path = disk_path
        self.prefix = prefix

        self.step_fn = step_fn
        self.log_to_wandb = bool(log_to_wandb)

        self._stop = threading.Event()
        self._t = None
        self._t0 = None
        self.samples = []

        self._nvml_inited = False
        self._nvml_handle = None

        if _NVML_OK and cuda_index is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_inited = True
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_index))
            except Exception:
                self._nvml_inited = False
                self._nvml_handle = None

    def start(self):
        ''' Starts background logging. '''
        if self._t is not None:
            return
        self._t0 = time.perf_counter()

        if _PSUTIL_OK:
            try:
                psutil.cpu_percent(interval=None)  # warm-up
            except Exception:
                pass

        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        ''' Stops background logging. '''
        if self._t is None:
            return
        self._stop.set()
        self._t.join(timeout=2.0)
        self._t = None
        self._stop.clear()

        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_inited = False

    def _gpu_stats(self):
        ''' Returns (util_pct, vram_used, vram_total) or (None, None, None). '''
        if not (self._nvml_inited and self._nvml_handle is not None):
            return None, None, None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return float(util), float(mem.used), float(mem.total)
        except Exception:
            return None, None, None

    def _loop(self):
        ''' Background logging loop '''
        while not self._stop.is_set():
            wall_s = time.perf_counter() - self._t0

            cpu_pct = ram_used = ram_total = ram_pct = None
            disk_used = disk_total = disk_pct = None

            if _PSUTIL_OK:
                try:
                    cpu_pct = safe_float(psutil.cpu_percent(interval=None))
                    vm = psutil.virtual_memory()
                    ram_used = safe_float(vm.used)
                    ram_total = safe_float(vm.total)
                    ram_pct = safe_float(vm.percent)

                    du = psutil.disk_usage(self.disk_path)
                    disk_used = safe_float(du.used)
                    disk_total = safe_float(du.total)
                    disk_pct = safe_float(du.percent)
                except Exception:
                    pass

            gpu_util, vram_used, vram_total = self._gpu_stats()

            row = {
                f"{self.prefix}_wall_time_s": wall_s,

                f"{self.prefix}_cpu_pct": cpu_pct,
                f"{self.prefix}_ram_used_gb": (ram_used / 1e9) if ram_used is not None else None,
                f"{self.prefix}_ram_total_gb": (ram_total / 1e9) if ram_total is not None else None,
                f"{self.prefix}_ram_pct": ram_pct,

                f"{self.prefix}_disk_used_gb": (disk_used / 1e9) if disk_used is not None else None,
                f"{self.prefix}_disk_total_gb": (disk_total / 1e9) if disk_total is not None else None,
                f"{self.prefix}_disk_pct": disk_pct,

                f"{self.prefix}_gpu_util_pct": gpu_util,
                f"{self.prefix}_vram_used_gb": (vram_used / 1e9) if vram_used is not None else None,
                f"{self.prefix}_vram_total_gb": (vram_total / 1e9) if vram_total is not None else None,
            }

            self.samples.append(row)
            if self.log_to_wandb:
                try:
                    step = self.step_fn() if self.step_fn is not None else None
                    if step is None:
                        # If you *must* log without step, then DO NOT use explicit steps elsewhere.
                        wandb.log(row, commit=False)
                    else:
                        wandb.log(row, step=int(step), commit=False)
                except Exception:
                    pass
                
            self._stop.wait(self.log_every_s)

    def aggregates(self) -> Dict[str, float]:
        """
            mean/max/p95 + duration_s for key metrics.
        """
        if not self.samples:
            return {}

        def _col(key: str) -> List[float]:
            vals = [r.get(key) for r in self.samples]
            vals = [v for v in vals if v is not None]
            return [float(v) for v in vals]

        out: Dict[str, float] = {}
        keys = [
            f"{self.prefix}_cpu_pct",
            f"{self.prefix}_ram_used_gb",
            f"{self.prefix}_ram_pct",
            f"{self.prefix}_disk_used_gb",
            f"{self.prefix}_disk_pct",
            f"{self.prefix}_gpu_util_pct",
            f"{self.prefix}_vram_used_gb",
        ]
        for k in keys:
            v = _col(k)
            if not v:
                continue
            arr = np.array(v, dtype=float)
            out[k + "_mean"] = float(arr.mean())
            out[k + "_max"] = float(arr.max())
            out[k + "_p95"] = float(np.percentile(arr, 95))

        wall = _col(f"{self.prefix}_wall_time_s")
        if wall:
            out[f"{self.prefix}_duration_s"] = float(max(wall))
        return out


def sum_profiler_flops(prof) -> Optional[float]:
    """
        Sum FLOPs across profiler key averages (best-effort).
        Returns FLOPs or None.
    """
    if prof is None:
        return None
    total = 0.0
    got_any = False
    try:
        for evt in prof.key_averages():
            fl = getattr(evt, "flops", None)
            if fl is not None and fl != 0:
                total += float(fl)
                got_any = True
    except Exception:
        return None
    return total if got_any else None


def train_step_with_optional_flops(
    enabled: bool,
    device: torch.device,
    do_profile_now: bool,
    train_fn,
    model,
    optimizer,
    data,
    **kwargs,
) -> Tuple[Optional[float], float]:
    """
        Executes the real train step exactly once, optionally wrapped by torch.profiler.
        Returns (flops, step_time_s). FLOPs may be None.
    """
    t0 = time.perf_counter()

    if enabled and do_profile_now and _PROFILER_OK:
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            acts.append(ProfilerActivity.CUDA)

        with torch_profile(
            activities=acts,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            train_fn(model, optimizer, data, **kwargs)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        flops = sum_profiler_flops(prof)
    else:
        train_fn(model, optimizer, data, **kwargs)
        flops = None
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    t1 = time.perf_counter()
    return flops, (t1 - t0)


def maybe_profile_macs_torchprofile(model, input_tensor) -> Optional[float]:
    """
        Optional MACs proxy using torchprofile.profile_macs(model, input_tensor).
        Returns MACs or None.
    """
    try:
        import torchprofile  # optional dependency
        model.eval()
        with torch.no_grad():
            macs = torchprofile.profile_macs(model, input_tensor)
        return float(macs)
    except Exception:
        return None
