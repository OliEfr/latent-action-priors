conda create --name latent_action_priors --file conda_requirements.txt
conda activate latent_action_priors
git clone git@github.com:OliEfr/loco-mujoco.git
cd loco-mujoco
git fetch --all
git checkout snake_swimmer
pip install .

