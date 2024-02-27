# generative-detection
Generative object detection using Variational Autoencoders (VAEs).

## Installation instructions

1. Clone the repository:
```bash
git clone REPOSITORY_URL
```
2. Navigate into the cloned repository:
```bash
cd generative-detection
```
3. Create a new conda environment with the provided environment file:
```bash
conda env create -f environment.yml
```
4. Activate the new conda environment:
```bash
conda activate gen-detection
```
5. Install the Python package:
```bash
pip install -e .
```

## Prepare ShapeNet Dataset
Please follow instructions in the [GET3D repository](https://github.com/nv-tlabs/GET3D/blob/master/render_shapenet_data/README.md) to download and render the ShapeNet dataset. Save this processed dataset at `ROOT/data/processed/shapenet/processed_get3d`.

The code we used to generate our ShapeNet dataset train, validation and test splits is in `src/data/datasets/shapenet.py` in the `create_splits` function. We set the `numpy` random seed as `23` to generate these splits.

## Train autoencoder on ShapeNet images
```bash
python train_autoencoder.py --base configs/autoencoder/train_shapenet/autoencoder_kl_8x8x64.yaml -t --gpus 0
```

## References
- Latent Diffusion Models: [ArXiv](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion)