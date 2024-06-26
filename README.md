# OD-VAE: Inverting Generation for 3D Object Detection
This repository is the official implementation of [OD-VAE: Inverting Generation for 3D Object Detection]().

> TODO: [Arxiv]() | [BibTex]()

> TODO: Method figure

[OD-VAE: Inverting Generation for 3D Object Detection]()
> TODO: [Anonymous Author(s)]()
> TODO: [Demo link]()

## Requirements

To install requirements:

1. Clone the repository:
```setup
git clone REPOSITORY_URL
```
2. Navigate into the cloned repository:
```setup
cd generative-detection
```
3. Create a new conda environment named `odvae` with the provided environment file:
```setup
conda env create -f environment.yml
```
4. Activate the new conda environment:
```setup
conda activate odvae
```

5. To initialize, fetch and checkout all the nested submodules:
```setup
git submodule update --init --recursive
```

6. Install the Python package:
```setup
pip install -e .
```

<!-- ## Prepare ShapeNet Dataset [4]
Please follow instructions in the [GET3D repository](https://github.com/nv-tlabs/GET3D/blob/master/render_shapenet_data/README.md) to download and render the ShapeNet dataset. Save this processed dataset at `ROOT/data/processed/shapenet/processed_get3d`.

The code we used to generate our ShapeNet dataset train, validation and test splits is in `src/data/datasets/shapenet.py` in the `create_splits` function. We set the `numpy` random seed as `23` to generate these splits. -->

## Prepare nuScenes Dataset [2]
> TODO

<!-- ## Prepare the Waymo Open Dataset [3]
> TODO -->

## Training

To train the model in the paper, run this command:
```train
srun python train.py -b configs/autoencoder/pose/autoencoder_kl_16x16x16.yaml -t --name od_vae_full_4gpu --devices 4
```

## Evaluation

To evaluate our model on nuScenes [2], run:
```eval
python eval.py --model-file odvae.pth --benchmark nuscenes
```
<!-- 
To evaluate our model on Waymo Open Dataset [3], run:
```eval
python eval.py --model-file odvae.pth --benchmark waymo
``` -->

## Pre-trained Models

You can download our pretrained model here:
- [OD-VAE]() trained on the nuScenes dataset [2] using parameters TODO. <!-- and the Waymo Open Dataset [3]  -->

If you use any of these models in your work, we are always happy to receive a [citation](CITATION.cff)
## Results

Our model achieves the following performance on :

### [3D Object Detection on nuScenes [2]](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes)

| Model name         | Metric 1        | Metric 2       |
| ------------------ |---------------- | -------------- |
| OD-VAE        |     xx%         |      xx%       |

<!-- ### [3D Object Detection on Waymo Open Dataset [3]](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle)

| Model name         | Metric 1        | Metric 2       |
| ------------------ |---------------- | -------------- |
| OD-VAE        |     xx%         |      xx%       | -->


## Contributing
The code in this repository is released under the [MIT License](LICENSE). We welcome any contributions to our repository via pull requests. 

## Comments
- Our codebase for the architecture of training of the VAE builds heavily on [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4). Thanks for open-sourcing!

## BibTeX
> TODO

## FAQ
- Downgrade MKL library to 2024.0.0 in case running `import torch` raises `undefined symbol: iJIT_NotifyEvent` from `torch/lib/libtorch_cpu.so`:
```bash
pip install mkl==2024.0.0
```

## References
[1] Latent Diffusion Models: [ArXiv](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion)

[2] nuScenes: [ArXiv](https://arxiv.org/abs/1903.11027)

<!-- [3] Waymo Open Dataset: [ArXiv](https://arxiv.org/abs/1912.04838)

[4] ShapeNet: [ArXiv](https://arxiv.org/abs/1512.03012) -->