import tensorflow as tf
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from SNN.old.env import RISEnvironment

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Non Spiking NN version
# Trying out continuous version to compare to discrete
class RISNormal(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, output_size)
        self.std = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.sigmoid(self.mu(x)) * (2 * torch.pi)
        std = F.softplus(self.std(x))
        return mu, std
    
    def get_dist(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        return dist
    
    def get_action(self, state, training=True):
        if training:
            dist = self.get_dist(state)
            action = dist.sample() % (2 * torch.pi)
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob
        else:
            mu, _ = self.forward(state)
            return mu


num_receivers = 1
ris_dims = [6, 6]
input_size = num_receivers * 2 + num_receivers # (x, y) + (rx_using_ris)
output_size = 36 # (ris_dims[0] * ris_dims[1] / variable)

net = RISNormal(input_size=input_size, output_size=output_size)
env = RISEnvironment(num_receivers=num_receivers, ris_dims=[ris_dims[0], ris_dims[1]])

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_steps = 20001
batch_size = 64

log_prob_batch = []
action_batch = []
reward_batch = []
state_batch = []

# Mean rewards for plotting
mean_rewards = []

mean_test_rewards = []
test_rx_pos = []
test_rx_using_ris = []
current_test = 0


for i in range(32):
    test_rx_pos.append(env.generate_rx_positions())
    test_rx_using_ris.append([1] + [random.choice([0, 1]) for _ in range(num_receivers - 1)])
    random.shuffle(test_rx_using_ris[i])

for step in range(num_steps):
    rx_pos = env.generate_rx_positions()
    env.update_rx_positions(rx_pos)

    # Generate dummy association matrix randomly
    rx_using_ris = [1] + [random.choice([0, 1]) for _ in range(num_receivers - 1)]
    random.shuffle(rx_using_ris)

    rx_pos_tensor = torch.tensor(rx_pos, dtype=torch.float32).flatten()
    rx_using_ris_tensor = torch.tensor(rx_using_ris, dtype=torch.float32).flatten()


    state = torch.cat((rx_pos_tensor, rx_using_ris_tensor)).unsqueeze(0)[None, :]

    action, log_prob = net.get_action(state)

    # phase_shifts = action.view(4, 8).repeat_interleave(2, dim=1)
  
    env.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(action), [1, ris_dims[0], ris_dims[1]])
    _, reward_real = env.calculate_reward(rx_using_ris)

    env.scene.get("ris").phase_profile.values = tf.zeros([1, ris_dims[0], ris_dims[1]])
    _, reward_baseline = env.calculate_reward(rx_using_ris)


    log_prob_batch.append(log_prob)
    action_batch.append(action)
    reward_batch.append((sum(reward_real) - sum(reward_baseline)) * 1000)
    state_batch.append(state)

    # Update the policy every batch_size steps
    if (step + 1) % batch_size == 0:

        rewards = torch.tensor(reward_batch, dtype=torch.float32)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized_rewards = (rewards - (-3)) / (3 - (-3))

        loss = -torch.stack(log_prob_batch) * normalized_rewards
        loss = loss.mean()
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}, Loss: {loss}")
        print(f"Mean Reward: {mean_reward*100}")
        
        mean_rewards.append(mean_reward)
        
        log_prob_batch = []
        action_batch = []
        reward_batch = []
        state_batch = []

    if step % 500 == 0:
        test_rewards = []
        for i in range(len(test_rx_pos)):
            env.update_rx_positions(test_rx_pos[i])
            
            state = torch.cat((torch.tensor(test_rx_pos[i], dtype=torch.float32).flatten(), torch.tensor(test_rx_using_ris[i], dtype=torch.float32).flatten())).unsqueeze(0)[None, :]
            action_test = net.get_action(state, training=False)
            # action_test = action_test.view(4, 8).repeat_interleave(2, dim=1).detach().numpy()
            
            env.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(action_test.detach().numpy()), [1, ris_dims[0], ris_dims[1]])
            _, reward = env.calculate_reward(test_rx_using_ris[i])
            env.scene.get("ris").phase_profile.values = tf.zeros([1, ris_dims[0], ris_dims[1]])
            _, reward_baseline = env.calculate_reward(test_rx_using_ris[i])
            test_rewards.append((sum(reward) - sum(reward_baseline)) * 10000)

        print("Test: ", np.mean(test_rewards))
        mean_test_rewards.append(np.mean(test_rewards))
        torch.save(net.state_dict(), f"snn_model{current_test}.pth")
        current_test += 1
        if current_test % 5 == 0:
            plt.plot(mean_test_rewards)
            plt.xlabel("Test")
            plt.ylabel("Mean Test Reward")
            plt.savefig(f"mean_test_rewards{current_test}.png")
            plt.clf()


# Plot the mean rewards over time
plt.plot(mean_rewards)
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.savefig("mean_rewards.png")
plt.clf()

plt.plot(mean_test_rewards)
plt.xlabel("Test")
plt.ylabel("Mean Test Reward")
plt.savefig("mean_test_rewards.png")