# Sequence-Aware Feature Training for Feature 3DGS

This repo is the **sequence-aware Python training extension for Feature 3DGS**, built on top of [`feature-3dgs`](https://github.com/yindaheng98/feature-3dgs) and [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting).
It extends the packaged Feature 3DGS Extractor-Decoder architecture from single scenes to multi-timestep 4D / dynamic-scene training.

Each timestep owns an independent `SemanticGaussianModel`, while the whole sequence shares a single learnable **Decoder** and a sequence-capable **Extractor**. This keeps per-frame Gaussian geometry separate while aligning all timesteps into one semantic feature space. Existing Feature 3DGS extractors can be reused through an inherent wrapper, and VGGT-based extractors can process all sequence images in one multi-view batch.

## Features

* [x] Organised as a standard Python package with `pip install` support
* [x] Sequence-aware Extractor-Decoder registry for 4D / multi-timestep training
* [x] Reuses all `feature_3dgs` extractors through `*-inherent` registrations
* [x] Shared decoder initialisation across all timesteps for a consistent feature space
* [x] All training modes inherited from Feature 3DGS: base, densify, camera, camera-densify

## Install

### Prerequisites

* [Pytorch](https://pytorch.org/) (>= v2.4 recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, match with PyTorch version)
* [gsplat](https://github.com/nerfstudio-project/gsplat)
* [`feature-3dgs`](https://github.com/yindaheng98/feature-3dgs)

Install `feature-3dgs` dependencies used by the inherited extractors:
```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/feature-3dgs.git@main --no-build-isolation
```

(Optional) If you have trouble with [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), try to install it from source:
```sh
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
```

## PyPI Install

```shell
pip install --upgrade feature-4dgs
```
or
build latest from source:
```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/feature-4dgs.git@master --no-build-isolation
```

### Development Install

```shell
git clone --recursive https://github.com/yindaheng98/feature-4dgs.git
cd feature-4dgs
pip install --target . --upgrade . --no-deps
```

### Download Checkpoints

Follow the checkpoint instructions in [`feature-3dgs`](https://github.com/yindaheng98/feature-3dgs). This package reuses the same inherited extractors and checkpoint layout.

## Command-Line Usage

### List Registered Extractor-Decoders

Verify that `feature_4dgs` can import and register sequence extractors:

```shell
python -c "import feature_4dgs; print(feature_4dgs.get_available_extractor_decoders())"
```

Every extractor registered by `feature_3dgs` is also available with an `-inherent` suffix, for example `dinov3_vitl16-inherent`.

### Train

```shell
python -m feature_4dgs.train \
    --name dinov3_vitl16-inherent --embed_dim 32 \
    -s data/sequence/frame_000 data/sequence/frame_001 data/sequence/frame_002 \
    -d output/sequence/frame_000-dinov3_vitl16 output/sequence/frame_001-dinov3_vitl16 output/sequence/frame_002-dinov3_vitl16 \
    -i 30000 \
    --mode densify \
    -e checkpoint_dir="'checkpoints'"
```

Each `-s/--sources` entry is one timestep's COLMAP / Gaussian Splatting scene directory, and each `-d/--destinations` entry is the matching output directory. The number of destinations must equal the number of sources.

### Resume From Saved Point Clouds

```shell
python -m feature_4dgs.train \
    --name dinov3_vitl16-inherent --embed_dim 32 \
    -s data/sequence/frame_000 data/sequence/frame_001 \
    -d output/sequence/frame_000-dinov3_vitl16 output/sequence/frame_001-dinov3_vitl16 \
    -l output/sequence/frame_000-dinov3_vitl16/point_cloud/iteration_30000/point_cloud.ply \
       output/sequence/frame_001-dinov3_vitl16/point_cloud/iteration_30000/point_cloud.ply \
    --load_decoder output/sequence/frame_000-dinov3_vitl16/point_cloud/iteration_30000/point_cloud.ply \
    -i 60000
```

The trainer saves each timestep independently under its destination directory, while `cameras.json` and semantic sidecar files follow the same layout as Feature 3DGS.

## API Usage

### Dataset & Decoder

```python
from feature_4dgs.prepare import prepare_datasets_and_decoder

datasets, decoder = prepare_datasets_and_decoder(
    name="dinov3_vitl16-inherent",   # registered sequence extractor-decoder name
    sources=[
        "data/sequence/frame_000",
        "data/sequence/frame_001",
    ],
    embed_dim=32,
    device="cuda",
    dataset_cache_device="cpu",
    configs={"checkpoint_dir": "checkpoints"},
)
# datasets is a SequenceFeatureCameraDataset; datasets[t] is a FeatureCameraDataset
# decoder is shared by all timesteps
```

### Gaussian Sequence

```python
from feature_4dgs.prepare import prepare_gaussians_sequence

gaussians_list = prepare_gaussians_sequence(
    decoder=decoder,
    sh_degree=3,
    sources=[
        "data/sequence/frame_000",
        "data/sequence/frame_001",
    ],
    datasets=datasets,
    device="cuda",
)
```

`prepare_gaussians_sequence` creates one `SemanticGaussianModel` per timestep. The first model initialises the shared decoder, then subsequent models load that decoder state so every frame starts in the same feature space.

### Training

```python
from feature_4dgs.train import prepare_training, training

datasets, gaussians_list, trainers = prepare_training(
    name="dinov3_vitl16-inherent",
    sh_degree=3,
    mode="densify",
    sources=["data/sequence/frame_000", "data/sequence/frame_001"],
    embed_dim=32,
    device="cuda",
    extractor_configs={"checkpoint_dir": "checkpoints"},
)
training(
    datasets=datasets,
    gaussians_list=gaussians_list,
    trainers=trainers,
    destinations=["output/frame_000-dinov3_vitl16", "output/frame_001-dinov3_vitl16"],
    iteration=30000,
    save_iterations=[7000, 30000],
)
```

### Inference

```python
import torch

with torch.no_grad():
    for dataset, gaussians in zip(datasets, gaussians_list):
        for camera in dataset:
            out = gaussians(camera)
            rgb = out["render"]                   # (3, H, W)
            feat = out["feature_map"]             # decoded, extractor-aligned
            feat_enc = out["feature_map_encoded"] # raw rasterised embeddings

        semantics = gaussians.get_semantics       # per-Gaussian semantic features
```

### Save & Load

```python
gaussians_list[0].save_ply("output/frame_000/point_cloud.ply")
# also saves point_cloud.ply.semantic.pt and point_cloud.ply.decoder.pt

gaussians_list[0].load_ply("output/frame_000/point_cloud.ply")
```

## Design: Sequence Extractor & Shared Decoder

The core abstraction extends Feature 3DGS by decoupling **how features are extracted across a sequence** from **how rasterised embeddings are decoded**.

### Sequence Extractor (`AbstractSequenceFeatureExtractor`)

The sequence extractor is a frozen foundation model wrapper that can process multiple timesteps. It inherits the single-image `AbstractFeatureExtractor` interface and adds `extract_sequence_all`:

```
Timestep image streams ──► Sequence Extractor (frozen) ──► Per-timestep feature maps
```

The default implementation simply calls `extract_all` once per timestep. Native sequence extractors may override this to batch or aggregate images across time.

### Shared Decoder (`AbstractTrainableDecoder`)

The decoder is the same trainable Feature 3DGS decoder shared by every timestep:

```
Frame 0 Gaussians ──┐
Frame 1 Gaussians ──┼──► Shared Decoder ──► Extractor-aligned feature maps
Frame 2 Gaussians ──┘
```

Sharing the decoder keeps all per-frame Gaussian embeddings aligned to a common feature space, while each timestep still has its own geometry, opacity, colour and encoded semantic tensors.

### Inherent Extractors

Any `feature_3dgs` extractor-decoder factory can be lifted into this sequence-aware package by wrapping its extractor with `InherentSequenceFeatureExtractor`. These registrations are suffixed with `-inherent`:

```shell
python -m feature_4dgs.train --name dinov3_vitl16-inherent --embed_dim 32 \
    -s data/frame_000 data/frame_001 \
    -d output/frame_000 output/frame_001
```

### Native VGGT Sequence Extractors

`VGGTSequenceExtractor` and `VGGTrackSequenceExtractor` flatten all timestep image streams into one call to VGGT's multi-view extractor, then split the result back per timestep. This lets VGGT use cross-view context over the full sequence before distillation starts.

## Extending: Adding a New Sequence Foundation Model

The project uses the same **auto-registration** pattern as Feature 3DGS. To add support for a new sequence model (e.g. a hypothetical `MyModel`), follow the VGGT implementation as a reference:

### Step 1: Implement the Sequence Extractor

Create `feature_4dgs/mymodel/extractor.py`:

```python
import torch
from feature_4dgs.extractor import AbstractSequenceFeatureExtractor

class MyModelSequenceExtractor(AbstractSequenceFeatureExtractor):
    def __init__(self, model, ...):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: (C, H, W) in [0, 1]
        # Return: (D, H', W') feature map
        ...

    def extract_sequence_all(self, sequences):
        # Optional override for cross-timestep batching or aggregation.
        ...

    def to(self, device) -> "MyModelSequenceExtractor":
        self.model.to(device)
        return self
```

### Step 2: Reuse or Implement the Decoder

Most models can reuse a `feature_3dgs` decoder such as `LinearDecoder`, or a model-specific decoder from Feature 3DGS. The key constraint is unchanged: `decode_feature_map` must output the same channel count and spatial size as the sequence extractor's feature maps.

### Step 3: Register via Factory

Create `feature_4dgs/mymodel/registry.py`:

```python
from feature_3dgs.decoder import LinearDecoder
from feature_4dgs.registry import register_extractor_decoder
from .extractor import MyModelSequenceExtractor

FEATURE_DIM = 768

def factory(embed_dim: int, **configs):
    extractor = MyModelSequenceExtractor(...)
    decoder = LinearDecoder(
        in_channels=embed_dim,
        out_channels=FEATURE_DIM,
    )
    return extractor, decoder

register_extractor_decoder("mymodel", factory)
```

### Step 4: Trigger Registration on Import

Create `feature_4dgs/mymodel/__init__.py`:

```python
from . import registry  # triggers register_extractor_decoder() at import time
```

Then add the import in `feature_4dgs/__init__.py`:

```python
from . import mymodel  # auto-registers "mymodel"
```

After these steps, the new model is available everywhere:

```shell
python -m feature_4dgs.train --name mymodel --embed_dim 32 \
    -s data/frame_000 data/frame_001 \
    -d output/frame_000-mymodel output/frame_001-mymodel \
    -i 30000
```

## Acknowledgement

This repo is developed based on [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), [feature-3dgs (packaged)](https://github.com/yindaheng98/feature-3dgs), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [gaussian-splatting (packaged)](https://github.com/yindaheng98/gaussian-splatting). Many thanks to the authors for open-sourcing their codebases.
