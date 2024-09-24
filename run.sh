# Before executing the experiments you need to setup a wandb account and specify your desired wandb entity here.
wandb_entity=YOUR_WANDB_ENTITY

# Baselines
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id AntCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HalfCheetahCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HumanoidCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id UnitreeA1Custom-v0
python train_H1.py --wandb_entity $wandb_entity 

# Style rewards
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id AntHasExpertDataCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HalfCheetahHasExpertDataCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HumanoidHasExpertDataCustom-v4
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id UnitreeA1HasExpertDataCustom-v0
python train_H1.py --wandb_entity $wandb_entity --use_expert_data

# Latent action priors
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id AntCustom-v4 --latent_action_space_dim 12 --projector_path expert_demonstrations/Ant_HalfCheetah/Ant-v4_decoder.pth --projector_size_0 4 --projector_size_1 8
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HalfCheetahCustom-v4 --latent_action_space_dim 9 --projector_path expert_demonstrations/Ant_HalfCheetah/HalfCheetah-v4_decoder.pth --projector_size_0 3 --projector_size_1 6
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HumanoidCustom-v4 --latent_action_space_dim 26 --projector_path expert_demonstrations/Humanoid/Humanoid_decoder.pth --projector_size_0 9 --projector_size_1 17
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id UnitreeA1Custom-v0 --latent_action_space_dim 18 --projector_path expert_demonstrations/Unitree_A1_H1/UnitreeA1.simple.perfect_decoder.pth --projector_size_0 6 --projector_size_1 12
python train_H1.py --wandb_entity $wandb_entity --latent_action_space_dim 17


# Latent action priors + style rewards
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id AntHasExpertDataCustom-v4 --latent_action_space_dim 12 --projector_path expert_demonstrations/Ant_HalfCheetah/Ant-v4_decoder.pth --projector_size_0 4 --projector_size_1 8
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HalfCheetahHasExpertDataCustom-v4 --latent_action_space_dim 9 --projector_path expert_demonstrations/Ant_HalfCheetah/HalfCheetah-v4_decoder.pth --projector_size_0 3 --projector_size_1 6
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id HumanoidHasExpertDataCustom-v4 --latent_action_space_dim 26 --projector_path expert_demonstrations/Humanoid/Humanoid_decoder.pth --projector_size_0 9 --projector_size_1 17
python train_Ant_HalfCheetah_Humanoid_A1.py --wandb_entity $wandb_entity --env_id UnitreeA1HasExpertDataCustom-v0 --latent_action_space_dim 18 --projector_path expert_demonstrations/Unitree_A1_H1/UnitreeA1.simple.perfect_decoder.pth --projector_size_0 6 --projector_size_1 12
python train_H1.py --wandb_entity $wandb_entity --use_expert_data --latent_action_space_dim 17

# 2xUnitree A1 envs
python train_2UnitreeA1.py --wandb_entity $wandb_entity # Baseline
python train_2UnitreeA1.py --wandb_entity $wandb_entity --env_id multiUnitreeA1ConnectedCustom-v0 # Style rewards
python train_2UnitreeA1.py --wandb_entity $wandb_entity --latent_action_space_dim 36 # Latent action priors
python train_2UnitreeA1.py --wandb_entity $wandb_entity --latent_action_space_dim 36 --env_id multiUnitreeA1ConnectedCustom-v0 # Latent action priors + style rewards


