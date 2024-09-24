import gymnasium as gym
import wandb
import sys
import time
import numpy as np
import torch
import tyro
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass
import envs
from wrappers import ProjectActions_multiUnitreeA1
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecVideoRecorder,
)


@dataclass
class Args:
    wandb_project_name: str = "latent_action_priors"
    wandb_entity: str = ""

    seed: int = 1
    env_id: str = "multiUnitreeA1ConnectedHasExpertDataCustom-v0"

    task_mode = "easy"  # easy

    latent_action_space_dim: int = (
        0  # Dimensionality of the action space used by the DRL policy. Set to 0 for baselines action space. Set to latent action space prior dimension for using only latent action space prior. Set to sum of baseline action space and latent action space prior for using projected actions and residual actions. For 2 Unitree A1 latent+full action space dim is 36.
    )
    projector_path: str = (
        "recorded_experts/loco_mujoco/loco_mujoco_a1.simple_latent_6_nonLin/UnitreeA1.simple.perfect_decoder.pth"
    )
    projector_size_0: int = 6
    projector_size_1: int = 12


args = tyro.cli(Args)

set_random_seed(args.seed)
args.device = "cuda" if torch.cuda.is_available() else "cpu"

# select training parameters
args.total_time_steps = 3_000_000
layer_size = int(np.sqrt(256 * 256 * 2))
nn_size = [layer_size, layer_size]
policy_kwargs = dict(net_arch=dict(pi=nn_size, vf=nn_size))
algo_kwargs = dict(n_steps=4 * 2048, batch_size=4 * 64)

# select residual weight for full action space
args.projector_residual_weight = 0.1

# perform checks
if args.latent_action_space_dim == False or args.latent_action_space_dim == 0:
    print("Using baseline action space.")
elif args.latent_action_space_dim == args.projector_size_0:
    print("Using only projected actions")
elif (
    args.latent_action_space_dim % (args.projector_size_0 + args.projector_size_1) == 0
):
    print(
        f"Using projected actions and residual actions with projector residual weight {args.projector_residual_weight}."
    )
else:
    raise ValueError(
        "The latent action space dimension should be equal to the sum of projected size and full action space (or a multiple of it for the (multiUnitreeA1))."
    )

assert (
    args.projector_residual_weight > 0 and args.projector_residual_weight <= 1
), "The residual weight must be between 0 and 1."


if __name__ == "__main__":

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        group="group",
        name="run",
        monitor_gym=True,
        save_code=True,
    )

    log_dir = f"./runs/{run.id}"

    def make_env():
        original_action_space = gym.make(
            args.env_id, latent_action_space_dim=False, n_unitreeA1=2
        ).action_space.shape[0]
        env = gym.make(
            args.env_id,
            n_unitreeA1=2,
            render_mode="rgb_array",
            latent_action_space_dim=args.latent_action_space_dim,
            mode=args.task_mode,
        )
        env = Monitor(env)
        if args.latent_action_space_dim:
            env = ProjectActions_multiUnitreeA1(
                env,
                action_space_shape=original_action_space,
                latent_action_space_dim=args.latent_action_space_dim,
                projector_path=args.projector_path,
                projector_size=[args.projector_size_0, args.projector_size_1],
                projector_type="nonLin",
                full_actions_weight=args.projector_residual_weight,
                n_UnitreeA1=2,
            )
        return env

    vec_env = DummyVecEnv([make_env for _ in range(1)])

    try:
        ep_length = vec_env.envs[0].env._max_episode_steps
    except:
        ep_length = vec_env.envs[0].env.env._max_episode_steps
    vec_env = VecVideoRecorder(
        vec_env,
        f"{log_dir}/videos",
        record_video_trigger=lambda x: x % (args.total_time_steps // 10) == 0,
        video_length=ep_length,
    )

    vec_env_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
    )
    eval_callback = EvalCallback(
        vec_env_norm,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/best_model_eval_results",
        eval_freq=(args.total_time_steps // 20),
        deterministic=True,
        render=False,
    )
    model = PPO(
        "MlpPolicy",
        vec_env_norm,
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        policy_kwargs=policy_kwargs,
        **algo_kwargs,
    )

    model.learn(
        total_timesteps=args.total_time_steps,
        callback=[
            eval_callback,
            WandbCallback(
                verbose=2,
                model_save_path=log_dir,
                model_save_freq=(args.total_time_steps // 20),
            ),
        ],
    )

    run.finish()
