import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import yaml
import os
from stable_baselines3.common.utils import set_random_seed
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import biplot, nonLinearAE, loss_function_AE_standardized, MAEloss

set_random_seed(0)

gym_env_ids = ["Ant-v4", "HalfCheetah-v4"]
results = {}

SAVE_ROOT_PATH = "./expert_demonstrations/Ant_HalfCheetah"
results = {}

if not os.path.exists(SAVE_ROOT_PATH):
    os.makedirs(SAVE_ROOT_PATH)

for gym_env_id in gym_env_ids:
    results[gym_env_id] = {}

    env = gym.make(gym_env_id)

    # load expert data
    expert_data = np.load(
        f"./expert_demonstrations/{gym_env_id}.npy",
        allow_pickle=True,
    ).item()
    expert_actions = expert_data["recorded_actions_single_cycle"]
    episode_actions = expert_data["recorded_actions_eps"]

    # conduct PCA and save results
    biplot(
        X=expert_actions,
        save_fig_path=f"{SAVE_ROOT_PATH}/{gym_env_id}",
    )

    # create ae model
    latent_dim = round(expert_actions.shape[1] / 2)
    AE_model = nonLinearAE(expert_actions.shape[1], latent_dim)

    # save number of samples and number of PCs in results file
    results[gym_env_id]["num_samples"] = expert_actions.shape[0]
    results[gym_env_id]["latent_dim"] = latent_dim

    # train ae model
    expert_actions_tensor = torch.tensor(expert_actions)
    dataset = TensorDataset(expert_actions_tensor, expert_actions_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    max_value_latent_space = torch.max(
        torch.abs(AE_model.encoder(episode_actions_tensor))
    )
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
    plt.legend()
    plt.savefig(f"{SAVE_ROOT_PATH}/{gym_env_id}_ae_training.pdf")
    plt.clf()

    # save MSE and MAE loss on train cycle and episode
    results[gym_env_id]["train cycle MSE loss"] = torch.nn.MSELoss()(
        AE_model(expert_actions_tensor), expert_actions_tensor
    ).item()
    results[gym_env_id]["train cycle MAE loss"] = MAEloss(
        AE_model(expert_actions_tensor), expert_actions_tensor
    ).item()
    results[gym_env_id]["full episode MSE loss"] = torch.nn.MSELoss()(
        AE_model(episode_actions_tensor), episode_actions_tensor
    ).item()
    results[gym_env_id]["full episode MAE loss"] = MAEloss(
        AE_model(episode_actions_tensor), episode_actions_tensor
    ).item()

    # save the decoder (used as prior in DRL)
    torch.save(
        AE_model.decoder.state_dict(),
        f"{SAVE_ROOT_PATH}/{gym_env_id}_decoder.pth",
    )

    # save the encoder
    torch.save(
        AE_model.encoder.state_dict(),
        f"{SAVE_ROOT_PATH}/{gym_env_id}_encoder.pth",
    )
# save results as yaml
with open(f"{SAVE_ROOT_PATH}/results.yaml", "w") as file:
    yaml.dump(results, file)
