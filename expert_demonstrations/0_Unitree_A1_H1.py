import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from moviepy.editor import ImageSequenceClip
import yaml
import gymnasium as gym
from loco_mujoco import LocoEnv
import loco_mujoco
import torch
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.common.utils import set_random_seed
from utils import (
    biplot,
    MAEloss,
    nonLinearAE,
    loss_function_AE_standardized,
)


# config to extract exactly one gait cycle
config = {
    "UnitreeA1.simple.perfect": {
        "start_frame": 150,
        "end_frame": 191,
        "seed": 1, # seed is chosen
        "latent_dim": 6,
        "save_root_path": "expert_demonstrations/Unitree_A1_H1",
    },
    "UnitreeH1.walk.perfect": {
        "start_frame": 212,
        "end_frame": 317,
        "seed": 0, # seed is chosen
        "latent_dim": 6,
        "save_root_path": "expert_demonstrations/Unitree_A1_H1",
    },
    # can add more environments here
}


SAVE_ROOT_PATH = None
results = {}

for env_name, conf in config.items():

    np.random.seed(conf["seed"])
    set_random_seed(conf["seed"])

    SAVE_ROOT_PATH = conf["save_root_path"]
    if not os.path.exists(SAVE_ROOT_PATH):
        os.makedirs(SAVE_ROOT_PATH)

    results[env_name] = {}

    env = gym.make(
        "LocoMujoco",
        env_name=env_name,
        render_mode="rgb_array",
        random_start=False,
        init_step_no=0,
    )

    # get expert data
    expert_dataset = env.create_dataset()
    expert_actions = expert_dataset["actions"]
    expert_states = expert_dataset["states"]

    recorded_obs = []
    recorded_relative_foot_positions = []

    # step through environment recording expert data
    imgs = []
    res_info = env.reset()
    recorded_obs.append(res_info[0])
    imgs.append(env.render().transpose(1, 0, 2))
    i = 0
    assert (
        conf["end_frame"] < 1001
    ), "if end_frame is too high -> rendering takes too long."
    while i < conf["end_frame"]:
        action = expert_actions[i, :]
        nstate, reward, terminated, truncated, info = env.step(action)
        recorded_obs.append(nstate)
        imgs.append(env.render().transpose(1, 0, 2))
        i += 1
    imgs = imgs[conf["start_frame"] : conf["end_frame"]]
    fps = 100  # int(1 / env.unwrapped.dt)
    clip = ImageSequenceClip(imgs, fps=fps)
    clip.write_gif(f"{SAVE_ROOT_PATH}/{env_name}.gif", fps=fps)

    # get expert data
    recorded_obs = np.array(recorded_obs)
    expert_actions = expert_actions[conf["start_frame"] : conf["end_frame"]]
    episode_actions = expert_dataset["actions"][:1000]
    expert_states = expert_states[conf["start_frame"] : conf["end_frame"]]
    episode_states = expert_dataset["states"][:1000]
    dict = {
        "recorded_actions_single_cycle": np.array(expert_actions),
        "recorded_states_single_cycle": np.array(expert_states),
        "recorded_obs_cycle": np.array(
            recorded_obs[conf["start_frame"] : conf["end_frame"]]
        ),  # recorded obs is indeed identical to expert_states
        "recorded_actions_eps": np.array(episode_actions),
        "recorded_states_eps": np.array(episode_states),
    }

    np.save(f"{SAVE_ROOT_PATH}/{env_name}.npy", dict)

    # conduct PCA and save results
    biplot(
        X=expert_actions,
        save_fig_path=f"{SAVE_ROOT_PATH}/{env_name}",
    )

    # create ae model
    latent_dim = config[env_name]["latent_dim"]
    AE_model = nonLinearAE(expert_actions.shape[1], latent_dim)

    # save number of samples and number of PCs in results file
    results[env_name]["num_samples"] = expert_actions.shape[0]
    results[env_name]["latent_dim"] = latent_dim

    # train data
    expert_actions_tensor = torch.tensor(expert_actions)
    dataset = TensorDataset(expert_actions_tensor, expert_actions_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # test data
    episode_actions_tensor = torch.tensor(episode_actions)

    optimizer = torch.optim.Adam(AE_model.parameters(), lr=3e-4, weight_decay=1e-8)

    epochs = 10_000
    mb_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(epochs):
        AE_model.train()
        mb_losses = []
        for data, _ in dataloader:

            latent = AE_model.encoder(data)
            reconstructed = AE_model.decoder(latent)

            loss = loss_function_AE_standardized(data, reconstructed, latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mb_losses.append(loss.item())
        epoch_train_losses.append(np.mean(mb_losses))
        with torch.no_grad():
            AE_model.eval()
            epoch_val_losses.append(
                loss_function_AE_standardized(
                    AE_model(episode_actions_tensor), episode_actions_tensor
                ).item()
            )
        print(f"Epoch {epoch+1}, loss: {np.mean(mb_losses)}")

    # latent space statistics
    max_value_latent_space = torch.max(
        torch.abs(AE_model.encoder(episode_actions_tensor))
    )
    if max_value_latent_space >= 1.0:
        results[env_name][
            "WARNING"
        ] = f"Latent space is not bounded by [-1, 1] for {max_value_latent_space}"
        print(
            f"WARNING: Latent space is not bounded by [-1, 1] for {max_value_latent_space}"
        )

    # plot training and validation loss
    matplotlib.rcParams.update({"font.size": 22})
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_train_losses, label="Training Loss")
    plt.plot(epoch_val_losses, label="Validation Loss")
    plt.savefig(f"{SAVE_ROOT_PATH}/{env_name}_ae_training.pdf")
    plt.legend()
    plt.clf()

    # save MSE and MAE loss on train cycle and episode
    results[env_name]["train cycle MSE loss"] = torch.nn.MSELoss()(
        AE_model(expert_actions_tensor), expert_actions_tensor
    ).item()
    results[env_name]["train cycle MAE loss"] = MAEloss(
        AE_model(expert_actions_tensor), expert_actions_tensor
    ).item()
    results[env_name]["full episode MSE loss"] = torch.nn.MSELoss()(
        AE_model(episode_actions_tensor), episode_actions_tensor
    ).item()
    results[env_name]["full episode MAE loss"] = MAEloss(
        AE_model(episode_actions_tensor), episode_actions_tensor
    ).item()

    # save the encoder and decoder. The decoder is used as a prior in DRL.
    torch.save(
        AE_model.decoder.state_dict(),
        f"{SAVE_ROOT_PATH}/{env_name}_decoder.pth",
    )
    torch.save(
        AE_model.encoder.state_dict(),
        f"{SAVE_ROOT_PATH}/{env_name}_encoder.pth",
    )

with open(f"{SAVE_ROOT_PATH}/results.yaml", "w") as file:
    yaml.dump(results, file)
