# Black-Box Privacy Attacks on Shared Representations in Multitask Learning

Code for the ICLR 2026 paper **Black-Box Privacy Attacks on Shared Representations in Multitask Learning**.

## Setup

First, install a Python package manager, then create and activate an environment.

**Conda**
```bash
conda create --name taskinf python=3.11
conda activate taskinf
```

**UV**
```bash
uv venv
source .venv/bin/activate
```

Then install dependencies using `pip` or `uv pip`:
```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

## Datasets

Download CelebA from [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) and save it in a directory ending in `celeba` (e.g. `/my/path/to/celeba/`). Use this full path as the `data_dir` parameter in the notebook.

## Usage

Open `celeba_personalization.ipynb` and set `data_dir` to your CelebA path, then run all cells.


## Datasets

Install CelebA at [this link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) and save it in a directory that ends in `celeba` (e.g. `/my/path/to/celeba/`). Use this full path as the `data_dir`
