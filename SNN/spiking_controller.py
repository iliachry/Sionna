import sionna as sn
import tensorflow as tf
import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
import matplotlib.pyplot as plt
import random

from env import RISEnvironment

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


# Define the RIS controller as an SNN
class RISController(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256, options_per_element = 4):
        super().__init__()
        
        self.spike_grad = surrogate.fast_sigmoid()   # Surrogate gradient for spiking
        self.options_per_element = options_per_element                 # Number of actions per RIS element
        self.lif = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)  # Leaky Integrate-and-Fire neuron
        self.T = 16
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.lif1 = self.lif

        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.lif2 = self.lif

        # Output layers for each RIS element
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, options_per_element) for _ in range(self.output_size)
        ])

        # LIF neurons for each RIS element
        self.lif_output = torch.nn.ModuleList([
            self.lif for _ in range(self.output_size)
        ])

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        output_mems = [lif.init_leaky() for lif in self.lif_output]
        
        spk_rec = [[] for _ in range(self.output_size)]

        # Generate spikes through SNN timesteps
        for _ in range(self.T):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            for i in range(self.output_size):
                spk_out, output_mems[i] = self.lif_output[i](self.output_layers[i](spk2), output_mems[i])
                spk_rec[i].append(spk_out)

        # Sum spikes per element across time steps 
        spike_counts = [torch.sum(torch.stack(spk, dim=1), dim=1) for spk in spk_rec]

        return spike_counts
    
    def get_action(self, state, is_training=True):
        spike_counts = self.forward(state)
        log_probs = [torch.nn.functional.log_softmax(counts.squeeze(0), dim=-1) for counts in spike_counts]
        actions = []
        action_log_probs = []
        if is_training:
            for i in range(self.output_size):
                probs = torch.exp(log_probs[i])
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                
                actions.append(action)
                action_log_probs.append(log_probs[i].gather(1, action.unsqueeze(1)))

            return actions, action_log_probs
        else:
            for i in range(self.output_size):
                probs = torch.exp(log_probs[i])
                action = torch.argmax(probs, dim=-1)

                actions.append(action)

            return actions
    
    # Convert tensor of actions to phase values
    def action_to_phase(self, action):
        action_array = np.array([a.item() for a in action])
        action_array = action_array.reshape(5, 5)
        action_array = (action_array / self.options_per_element) * (2 * np.pi)
        action_array = np.repeat(np.repeat(action_array, 2, axis=0), 2, axis=1)
        return action_array
    
    # Compute new log probabilities for the actions taken
    # Required for PPO training objective
    def compute_new_log_probs_and_entropy(self, states, actions):
        spike_counts = self.forward(states)
        log_probs = [torch.nn.functional.log_softmax(counts.squeeze(0), dim=-1) for counts in spike_counts]
        new_log_probs = []
        entropies = []
        
        for i in range(self.output_size):
            # Get log probs for the chosen actions
            log_probs_i = log_probs[i].squeeze(1)
            new_log_probs.append(log_probs_i.gather(1, actions[:, i].unsqueeze(1)))
            # new_log_probs.append(log_probs[i].gather(1, actions[i].unsqueeze(1)))
            
            # Compute entropy
            probs = torch.exp(log_probs[i])  
            entropy = -torch.sum(probs * log_probs[i], dim=1)  
            entropies.append(entropy)  
        # Stack all entropies and take the mean
        entropy_bonus = torch.mean(torch.stack(entropies))
        new_log_probs = torch.cat(new_log_probs, dim=1).sum(dim=1)


        return new_log_probs, entropy_bonus


# Initialize the RIS controller and optimizer
num_receivers = 1
ris_dims = [10, 10]
input_size = num_receivers * 2 + num_receivers # (x, y) + (rx_using_ris)
output_size = 25 # (ris_dims[0] * ris_dims[1]) / 4

net = RISController(input_size=input_size, output_size=output_size)
env = RISEnvironment(num_receivers=num_receivers, ris_dims=ris_dims)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Training loop
num_steps = 15001
batch_size = 128

log_prob_batch = []
action_batch = []
reward_batch = []
state_batch = []
entropy_batch = []

# Mean rewards for plotting
mean_rewards = []

mean_test_rewards = []
test_rx_pos = []
test_rx_using_ris = []
current_test = 0


for i in range(20):
    test_rx_pos.append(env.generate_rx_positions())
    test_rx_using_ris.append([1] + [random.choice([0, 1]) for _ in range(num_receivers - 1)])
    random.shuffle(test_rx_using_ris[i])


for step in range(num_steps):
    if step % 4 == 0:
        rx_pos = env.generate_rx_positions()
        env.update_rx_positions(rx_pos)

    # Generate dummy association matrix randomly
    rx_using_ris = [1] + [random.choice([0, 1]) for _ in range(num_receivers - 1)]
    random.shuffle(rx_using_ris)
    
    rx_pos_tensor = torch.tensor(rx_pos, dtype=torch.float32).flatten()
    rx_using_ris_tensor = torch.tensor(rx_using_ris, dtype=torch.float32).flatten()

    state = torch.cat((rx_pos_tensor, rx_using_ris_tensor)).unsqueeze(0)[None, :]
    action, log_probs_per_element = net.get_action(state, is_training=True)
    log_prob = torch.stack(log_probs_per_element).sum()

    phase_shift = net.action_to_phase(action)
    env.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, ris_dims[0], ris_dims[1]])
    _, reward_real = env.calculate_reward(rx_using_ris)

    env.scene.get("ris").phase_profile.values = tf.zeros([1, ris_dims[0], ris_dims[1]])
    _, reward_baseline = env.calculate_reward(rx_using_ris)

    log_prob_batch.append(log_prob)
    action_batch.append(action)
    reward_batch.append((sum(reward_real) - sum(reward_baseline)) * 1000)
    state_batch.append(state)

    # Update the policy every batch_size steps
    if (step + 1) % batch_size == 0:

        old_log_probs = torch.stack(log_prob_batch).detach()
        rewards = torch.tensor(reward_batch, dtype=torch.float32)

        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-6)

        for i in range(4):
            
            new_log_probs, entropy = net.compute_new_log_probs_and_entropy(torch.cat(state_batch), torch.stack([torch.tensor(a) for a in action_batch]))
            ratio = torch.exp(new_log_probs - old_log_probs)
            epsilon = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            ppo_loss = -torch.min(ratio * normalized_rewards, clipped_ratio * normalized_rewards).mean()
            loss = ppo_loss - 0.01 * torch.mean(entropy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Step {step + 1}, Loss: {loss}")
        print(f"Mean Reward: {mean_reward*10}")

        mean_rewards.append(mean_reward)
        
        log_prob_batch = []
        action_batch = []
        reward_batch = []
        state_batch = []
        entropy_batch = []
  
    if step % 1000 == 0:
        test_rewards = []
        for i in range(len(test_rx_pos)):
            env.update_rx_positions(test_rx_pos[i])
            state = torch.cat((torch.tensor(test_rx_pos[i], dtype=torch.float32).flatten(), torch.tensor(test_rx_using_ris[i], dtype=torch.float32).flatten())).unsqueeze(0)[None, :]
            action = net.get_action(state, is_training=False)
            phase_shift = net.action_to_phase(action)
            env.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, ris_dims[0], ris_dims[1]])
            _, reward = env.calculate_reward(test_rx_using_ris[i])
            test_rewards.append(sum(reward))

        print("Test: ", np.mean(test_rewards))
        mean_test_rewards.append(np.mean(test_rewards))
        torch.save(net.state_dict(), f"snn_model{current_test}.pth")
        current_test += 1
    

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