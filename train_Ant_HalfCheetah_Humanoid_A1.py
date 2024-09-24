import gymnasium as gym
import wandb
import numpy as np
import torch
import tyro
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass
import envs
from wrappers import ProjectActions
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
    env_id: str = "AntCustom-v4"

    latent_action_space_dim: int = (
        0  # Dimensionality of the action space used by the DRL policy. Set to 0 for baselines action space. Set to latent action space prior dimension for using only latent action space prior. Set to sum of baseline action space and latent action space prior for using projected actions and residual actions.
    )

    projector_path: str = (
        ""  # path to the decoder for the latent action prior. Only used if latent_action_space_dim != 0
    )

    projector_size_0: int = 9  # Dimensionality of the latent action space prior
    projector_size_1: int = (
        17  # Dimensionality of the output of the latent action space prior decoder. Should be equal to baseline action space
    )

    target_speed_scale: float = (
        1.0  # only for UnitreeA1Custom-v0 and UnitreeA1HasExpertDataCustom-v0 to set the target speed factors
    )
    mode: str = (
        "easy"  # only for UnitreeA1Custom-v0 and UnitreeA1HasExpertDataCustom-v0 to set the mode. Hard = sample any target direction.
    )


args = tyro.cli(Args)

set_random_seed(args.seed)
args.device = "cuda" if torch.cuda.is_available() else "cpu"

# select training parameters based on environment
args.total_time_steps = 2_000_000
algo_kwargs = {}
policy_kwargs = {}
if args.env_id in [
    "HumanoidCustom-v4",
    "HumanoidHasExpertDataCustom_v4",
]:
    policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))
    args.total_time_steps = 5_000_000
    algo_kwargs = dict(n_steps=4 * 2048, batch_size=4 * 64, ent_coef=0.01)

if args.env_id in ["UnitreeA1Custom-v0", "UnitreeA1HasExpertDataCustom-v0"]:
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    algo_kwargs = dict(n_steps=2 * 2048, batch_size=2 * 64)


# select residual weight for full action space
args.full_actions_weight = 0.1
if args.env_id in ["HumanoidCustom-v4", "HumanoidHasExpertDataCustom_v4"]:
    args.full_actions_weight = 0.5
if "UnitreeA1" in args.env_id:
    if args.target_speed_scale == 3.0:
        args.full_actions_weight = 0.3
    if args.target_speed_scale == 4.0:
        args.full_actions_weight = 0.5
    if args.mode == "hard":
        args.full_actions_weight = 0.2

# perform checks
if args.latent_action_space_dim == False or args.latent_action_space_dim == 0:
    print("Using baseline action space.")
elif args.latent_action_space_dim == args.projector_size_0:
    print("Using only projected actions.")
elif args.latent_action_space_dim == (args.projector_size_0 + args.projector_size_1):
    print("Using projected actions and residual actions.")
else:
    raise ValueError(
        "The latent action space dimension should be equal to the sum of projected size and full action space."
    )

assert (
    args.full_actions_weight > 0 and args.full_actions_weight <= 1
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
            args.env_id, latent_action_space_dim=False
        ).action_space.shape[0]
        env_args = {}
        if args.env_id in ["UnitreeA1Custom-v0", "UnitreeA1HasExpertDataCustom-v0"]:
            env_args["target_speed_scale"] = args.target_speed_scale
            env_args["mode"] = args.mode
        env = gym.make(
            args.env_id,
            render_mode="rgb_array",
            latent_action_space_dim=args.latent_action_space_dim,
            **env_args,
        )
        env = Monitor(
            env,
        )
        if args.latent_action_space_dim:
            env = ProjectActions(
                env,
                action_space_shape=original_action_space,
                latent_action_space_dim=args.latent_action_space_dim,
                projector_path=args.projector_path,
                projector_type="nonLin",
                projector_size=[args.projector_size_0, args.projector_size_1],
                full_actions_weight=args.full_actions_weight,
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
