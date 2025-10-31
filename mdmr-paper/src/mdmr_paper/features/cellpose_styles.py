from pathlib import Path
import numpy as np
from cellpose import models

def extract_or_load_styles(
    images,
    *, device, model_type, channels,
    cache_dir: str, cache_key: str,
    overwrite: bool = False
) -> np.ndarray:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{cache_key}.npy"
    if path.exists() and not overwrite:
        return np.load(path)
    model = models.Cellpose(device=device, gpu=True, model_type=model_type)
    _, _, styles, _ = model.eval(images, diameter=None, flow_threshold=None, channels=channels, progress=True)
    X = np.asarray(styles)  # shape (N, 256)
    np.save(path, X)
    return X
