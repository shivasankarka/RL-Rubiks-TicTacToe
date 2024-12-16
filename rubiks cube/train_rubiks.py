import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim

import numpy as np
import random

from rubiks2x2 import Rubiks2x2


INPUT_DIMS = 24
OUTPUT_DIMS = 1

class CubeNeuralNetwork(nn.Module):
    def __init__(self, device):
        super(CubeNeuralNetwork, self).__init__()
        self.device = device
        # Convolutional Layers
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # Conv1
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # Pool1 (12x2 -> 6x1)
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # Conv2
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 1)),  # Pool2 (6x1 -> 3x1)
        # )

        # # Fully Connected (MLP) Layers
        # self.fc_layers = nn.Sequential(
        #     nn.Flatten(),  # Replace manual reshape
        #     nn.Linear(32 * 3 * 1, 128),  # Adjust dimensions based on Conv2d output size
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)  # Predict a single scalar: number of moves
        # )
        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            # First Convolutional Layer (6 -> 32 channels)
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),

            # Second Convolutional Layer (32 -> 64 channels)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),

            # Third Convolutional Layer (64 -> 128 channels)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(),

            # Fourth Convolutional Layer (128 -> 256 channels)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(),

            # Fifth Convolutional Layer (256 -> 512 channels)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.ReLU(),

            # MaxPool Layer (optional, can help reduce dimensionality in a more complex model)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pool1 (12x2 -> 6x1)
        )

        # Fully Connected (MLP) Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output from conv layers
            nn.Linear(512 * 6 * 1, 2048),  # Adjust based on the new output size
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Ensure input is correctly shaped
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        x.to(self.device)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

moves = ["R", "Rp", "L", "Lp", "U", "Up", "D", "Dp", "F", "Fp", "B", "Bp"]
batch_size = 128

class CubeSolver():
    def __init__(self, learning_rate: float = 1e-3):
        self.env = Rubiks2x2()
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print("Using {} backend".format(self.device))
        self.nn = CubeNeuralNetwork(self.device).to(self.device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def train(self, num_ep: int = 20000):
        for ep in range(num_ep):
            states = np.zeros((batch_size, 12, 2, 6), dtype=np.float32)  # (batch_size, 12, 2, 6)
            true_move_counts = np.zeros(batch_size, dtype=np.float32)  # (batch_size,)
            for i in range(batch_size):
                # for _ in range(5):
                moves_sample = random.randint(1, 3)
                # moves_sample = 1 +  int(ep / 1000.0)
                # print("Moves sample: ", moves_sample)
                scramble = [str(random.choice(moves)) for _ in range(moves_sample)]
                self.env.scramble(scramble)
                states[i] = self.env.state
                true_move_counts[i] = moves_sample

            states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2)
            true_move_counts = torch.tensor(true_move_counts, dtype=torch.float32).unsqueeze(1)

            states = states.contiguous()
            true_move_counts = true_move_counts.contiguous()

            states = states.to(self.device)
            true_move_counts = true_move_counts.to(self.device)

            predictions = self.nn(states)
            # predictions_rounded = torch.round(predictions)
            loss = self.loss_function(predictions, true_move_counts)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if ep % 100 == 0:
                # for param in self.nn.parameters():
                    # print(f"Gradient: {param.grad}")
                # print("predict: ", self.nn(states[0]), "true: ", true_move_counts[0])
                print(f"Episode {ep}: Loss = {loss.item()}")

if __name__ == "__main__":
    # Create and train the agent
    agent = CubeSolver()
    agent.train()

    torch.save(agent.nn.state_dict(), 'model_weights.pth')

    # Example prediction
    env = Rubiks2x2()
    env.scramble(["R"])
    scrambled_tensor = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(agent.device)
    predicted_moves = agent.nn(scrambled_tensor)
    print(f"Predicted moves to solve: {predicted_moves}")
