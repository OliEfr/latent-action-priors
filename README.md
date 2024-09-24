### Code accompanying the latent action priors paper

🖥️ [Website with videos](https://sites.google.com/view/latent-action-priors) | 📄 [Paper]()

**How to use this repository**: This repository uses `conda` to manage the python environment. Run `source setup.sh` to setup the repository. `expert_demonstrations` contains the data and code to reproduce the single gait cycle expert demonstrations used in the paper. Files with `train_*` run the trainings conducted in the paper. For examples on how to use these training files and conduct experiments see `run.sh`. If you are getting started with this repository, you probably want to look at `run.sh`.

**Important**: You require a `wandb` account to log the experiments in the repository. Before you run the `train_*.py` files set your desired `wandb-entity` either in those files or in `run.sh`.


**Note**: The recorded `.gif` videos of the agents sometimes show incorrect playback speeds as `.gif` only supports framerates of up to 50fps for most browsers. The video playback speeds on the accompanying website are correct.

Cite the paper and project as follows:
> cite

ToDo
- fix multi unitreeA1 envs