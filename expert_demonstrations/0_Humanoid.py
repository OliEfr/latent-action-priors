import gymnasium as gym
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
import torch
from sb3_contrib import TQC
from stable_baselines3.common.utils import set_random_seed
import torch
from torch.utils.data import TensorDataset, DataLoader
from moviepy.editor import ImageSequenceClip
from utils import (
    biplot,
    MAEloss,
    nonLinearAE,
    loss_function_AE_standardized,
)


# config to extract one gait cycle
config = {
    "Humanoid": {
        "env_id": "Humanoid-v4",
        "fps": 67,
        "start_frame": 72, # begining of gait cycle
        "end_frame": 94, # end of gait cycle
        "model": TQC,
        "expert_policy": "./expert_demonstrations/Humanoid.zip",
        "latent_dim": 9,
        "save_root_path": "./expert_demonstrations",
    },
    # can add other DRL experts here
}

model_name = "Humanoid"

results = {}
results[model_name] = {}

SAVE_ROOT_PATH = config[model_name]["save_root_path"] + f"/{model_name}"
if not os.path.exists(SAVE_ROOT_PATH):
    os.makedirs(SAVE_ROOT_PATH)

np.random.seed(1)
set_random_seed(1)

# create env
env = gym.make(
    config[model_name]["env_id"],
    render_mode="rgb_array",
)
env.reset(seed=1)

# load expert
model = config[model_name]["model"].load(
    config[model_name]["expert_policy"],
    env=env,
)

# step through env and log expert data
video = []
vec_env = model.get_env()
obs = vec_env.reset()
vec_env.seed(seed=1)
record_n_steps = config[model_name]["end_frame"]
recorded_obs = np.zeros((record_n_steps + 1, obs.shape[1]))
recorded_actions = np.zeros((record_n_steps, vec_env.action_space.shape[0]))

recorded_obs[0] = obs
for i in range(record_n_steps):
    action, _ = model.predict(obs, deterministic=True)
    recorded_actions[i] = action
    obs, rewards, dones, info = vec_env.step(action)
    recorded_obs[i + 1] = obs
    pixels = vec_env.envs[0].render()
    video.append(pixels)
    if dones:
        raise ValueError("Episode should not terminate.")

# save video
clip = ImageSequenceClip(
    video[config[model_name]["start_frame"] : config[model_name]["end_frame"]],
    fps=config[model_name]["fps"],
)
clip.write_gif(f"{SAVE_ROOT_PATH}/{model_name}.gif", fps=config[model_name]["fps"])

# save expert data
episode_actions = recorded_actions
expert_actions = recorded_actions[
    config[model_name]["start_frame"] : config[model_name]["end_frame"]
]
dict = {
    "recorded_actions_single_cycle": np.array(expert_actions),
    "recorded_actions_eps": np.array(episode_actions),
    "recorded_obs_cycle": np.array(
        recorded_obs[
            config[model_name]["start_frame"] : config[model_name]["end_frame"]
        ]
    ),
}

np.save(f"{SAVE_ROOT_PATH}/{model_name}.npy", dict)

# conduct PCA and save results
biplot(
    X=expert_actions,
    save_fig_path=f"{SAVE_ROOT_PATH}/{model_name}",
)

# create ae model
latent_dim = config[model_name]["latent_dim"]
AE_model = nonLinearAE(expert_actions.shape[1], latent_dim)

# save number of samples and number of PCs in results file
results[model_name]["num_samples"] = expert_actions.shape[0]
results[model_name]["latent_dim"] = latent_dim

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
max_value_latent_space = torch.max(torch.abs(AE_model.encoder(episode_actions_tensor)))
if not max_value_latent_space < 1.0:
    print(
        f"WARNING: Latent space is not bounded by [-1, 1] for {max_value_latent_space}"
    )

# plot training and validation loss
matplotlib.rcParams.update({"font.size": 22})
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_train_losses, label="Training Loss")
plt.plot(epoch_val_losses, label="Validation Loss")
plt.savefig(f"{SAVE_ROOT_PATH}/{model_name}_ae_training.pdf")
plt.legend()
plt.clf()

# save MSE and MAE loss on train cycle and episode
results[model_name]["train cycle MSE loss"] = torch.nn.MSELoss()(
    AE_model(expert_actions_tensor), expert_actions_tensor
).item()
results[model_name]["train cycle MAE loss"] = MAEloss(
    AE_model(expert_actions_tensor), expert_actions_tensor
).item()
results[model_name]["full episode MSE loss"] = torch.nn.MSELoss()(
    AE_model(episode_actions_tensor), episode_actions_tensor
).item()
results[model_name]["full episode MAE loss"] = MAEloss(
    AE_model(episode_actions_tensor), episode_actions_tensor
).item()

# save the encoder and decoder. The decoder is later used as a prior in DRL.
torch.save(
    AE_model.decoder.state_dict(),
    f"{SAVE_ROOT_PATH}/{model_name}_decoder.pth",
)
torch.save(
    AE_model.encoder.state_dict(),
    f"{SAVE_ROOT_PATH}/{model_name}_encoder.pth",
)

with open(f"{SAVE_ROOT_PATH}/results.yaml", "w") as file:
    yaml.dump(results, file)
