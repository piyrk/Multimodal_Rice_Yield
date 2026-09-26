"""CPU/CUDA forward-pass smoke test for the multimodal architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import RiceMultimodalDataset, multimodal_collate
from models.fusion_model import MultimodalYieldModel


ROOT = Path(__file__).resolve().parents[2]


def parameter_report(model: torch.nn.Module) -> tuple[int, float]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    size_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (
        1024**2
    )
    return parameters, size_mb


def run_device(model: torch.nn.Module, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    model = model.to(device).eval()
    inputs = {
        key: batch[key].to(device)
        for key in ("satellite", "weather", "soil", "weather_mask")
    }
    with torch.inference_mode():
        details = model(**inputs, return_details=True)
    for name, value in details.items():
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains NaN/Inf on {device}")
    return {key: value.cpu() for key, value in details.items()}


def main() -> int:
    dataset = RiceMultimodalDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=multimodal_collate)
    batches = []
    for index, batch in enumerate(loader):
        batches.append(batch)
        if index == 2:
            break
    if len(batches) != 3:
        raise AssertionError("Expected three batches for the forward smoke test")
    model = MultimodalYieldModel()
    parameters, size_mb = parameter_report(model)
    cpu_results = [run_device(model, batch, torch.device("cpu")) for batch in batches]
    cpu_details = cpu_results[0]
    print(f"satellite embedding shape = {tuple(cpu_details['satellite_embedding'].shape)}")
    print(f"weather embedding shape = {tuple(cpu_details['weather_embedding'].shape)}")
    print(f"soil embedding shape = {tuple(cpu_details['soil_embedding'].shape)}")
    print(f"fused representation shape = {tuple(cpu_details['fused'].shape)}")
    print(f"attention weights shape = {tuple(cpu_details['attention_weights'].shape)}")
    print(f"prediction shape = {tuple(cpu_details['prediction'].shape)}")
    print(f"prediction finite = {bool(torch.isfinite(cpu_details['prediction']).all())}")
    print(f"trainable parameters = {parameters}")
    print(f"approximate parameter size MB = {size_mb:.3f}")
    print("CPU forward pass = PASSED")
    if torch.cuda.is_available():
        run_device(model, batches[0], torch.device("cuda"))
        print("CUDA forward pass = PASSED")
    else:
        print("CUDA forward pass = SKIPPED (CUDA unavailable)")
    print("multiple batches loaded = PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
