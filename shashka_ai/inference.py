""" Inference qatlami: PyTorch checkpoint YOKI ONNX modeldan yagona eval_fn.
eval_fn(x [B,194] float32) -> (policy_logits [B,1024], values [B])
bu yerda value = P(win) - P(loss)  (WDL softmax dan). """

from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple
EvalFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

def wdl_to_value(wdl_logits: np.ndarray) -> np.ndarray:
    z = wdl_logits - wdl_logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return (p[..., 0] - p[..., 2]).astype(np.float32)

def load_ckpt(path: str) -> Dict[str, Any]:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def build_network_from_ckpt(path: str):
    from network import ShashkaNet
    ckpt = load_ckpt(path)
    net = ShashkaNet(**ckpt["hparams"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt

class TorchEvaluator:
    def __init__(self, net, device: str = "cpu", use_amp: bool = False) -> None:
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, wdl = self.net(t)
            else:
                logits, wdl = self.net(t)
            logits = logits.float().cpu().numpy()
            wdl = wdl.float().cpu().numpy()
        return logits, wdl_to_value(wdl)

class OnnxEvaluator:
    def __init__(self, path: str) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits, wdl = self.sess.run(["policy_logits", "wdl_logits"], {"x": np.ascontiguousarray(x, dtype=np.float32)})
        return logits, wdl_to_value(wdl)

def build_eval_fn(ckpt: Optional[str] = None, onnx: Optional[str] = None, device: str = "cpu") -> EvalFn:
    if onnx:
        return OnnxEvaluator(onnx)
    if ckpt:
        net, _ = build_network_from_ckpt(ckpt)
        return TorchEvaluator(net, device=device)
    raise ValueError("Model ko'rsatilmadi: --ckpt yoki --onnx bering")