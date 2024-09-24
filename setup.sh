conda env create -f environment.yml
conda activate latent-action-priors
git clone git@github.com:OliEfr/loco-mujoco.git
cd loco-mujoco
git checkout snake_swimmer
pip install .
loco-mujoco-download # we require the loco-mujoco datasets to optimally reuse the loco-mujoco benchmark. Datasets are a few GB and take some time to download.
