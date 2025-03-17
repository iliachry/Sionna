import sionna as sn
import tensorflow as tf
import numpy as np
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
sn.config.seed = 42
torch.manual_seed(42)

# Define the scene and components
scene = sn.rt.load_scene(sn.rt.scene.simple_street_canyon)

scene.tx_array = sn.rt.PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")

scene.rx_array = sn.rt.PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")

tx = sn.rt.Transmitter(name="tx", position=[-32,10,32])
scene.add(tx)

rx_positions = [[22,52,1.7], [25,52,1.7], [28,52,1.7], [31,52,1.7]]
rx = sn.rt.Receiver(name="rx", position=[22,52,1.7])
scene.add(rx)

ris_position = [32,-9,32]   # Positioned between Tx and Rx
num_elements = 9            # 3x3 grid of RIS elements
num_rows = 3
num_cols = 3
ris = sn.rt.RIS(name="ris1", position=ris_position, num_rows=num_rows, num_cols=num_cols, look_at=(tx.position+rx.position) / 2)
scene.add(ris)

camera = sn.rt.Camera("Cam", [0, 0, 300], look_at=[0, 0, 0])
scene.add(camera)

initial_phases = tf.random.uniform([1, num_rows, num_cols], minval=0, maxval=2*np.pi)

cell_grid = sn.rt.CellGrid(num_rows, num_cols)
ris.phase_profile = sn.rt.DiscretePhaseProfile(cell_grid=cell_grid, num_modes=1, values=initial_phases)


input_size = 3                          # receiver position (x, y, z)
hidden_size = 32                        # Hidden layer size
output_size = num_elements              # Output: phase adjustments for each RIS element
spike_grad = surrogate.fast_sigmoid()   # Surrogate gradient for spiking
options_per_element = 4                 # Number of actions per RIS element
lif = snn.Leaky(beta=0.9, spike_grad=spike_grad)  # Leaky Integrate-and-Fire neuron
T = 16                                  # Number of time steps for the SNN

# Define the RIS controller as an SNN
class RISController(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.T = T
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.lif1 = lif

        # Output layers for each RIS element
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, options_per_element) for _ in range(num_elements)
        ])

        # LIF neurons for each RIS element
        self.lif_output = torch.nn.ModuleList([
            lif for _ in range(num_elements)
        ])

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        output_mems = [lif.init_leaky() for lif in self.lif_output]
        
        spk_rec = [[] for _ in range(num_elements)]

        # Generate spikes through SNN timesteps
        for _ in range(self.T):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            for i in range(num_elements):
                spk_out, output_mems[i] = self.lif_output[i](self.output_layers[i](spk1), output_mems[i])
                spk_rec[i].append(spk_out)

        # Sum spikes per element across time steps 
        spike_counts = [torch.stack(spk, dim=0).sum(dim=0) for spk in spk_rec]

        return spike_counts
    
    def get_action(self, state):
        spike_counts = self.forward(state)

        log_probs = [torch.nn.functional.log_softmax(counts.squeeze(0), dim=-1) for counts in spike_counts]

        actions = []
        action_log_probs = []

        for i in range(num_elements):
            probs = torch.exp(log_probs[i])
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            
            actions.append(action)
            action_log_probs.append(log_probs[i].gather(1, action.unsqueeze(1)))

        return actions, action_log_probs


# Utility function to compute received power
def compute_received_power():
    paths = scene.compute_paths(max_depth=2)  # Compute paths with up to 2 reflections
    
    # Uncomment to visualize scene
    # sn.rt.Scene.render_to_file(self=scene, camera="Cam", paths=paths, filename="scene.png")

    a, tau = paths.cir() 
    a = torch.tensor(a.numpy(), dtype=torch.complex64, requires_grad=True)

    return torch.sum(torch.abs(a) ** 2)  # Sum of squared magnitudes = received power

# Convert tensor of actions to phase values
def action_to_phase(action):
    action_array = np.array([a.item() for a in action])
    return (action_array / 4) * (2 * np.pi)


# Initialize the RIS controller and optimizer
net = RISController()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_steps = 2000
batch_size = 32

log_prob_batch = []
action_batch = []
reward_batch = []
state_batch = []

# Mean rewards for plotting
mean_rewards = []

for step in range(num_steps):
    # Simulate receiver movement by randomly selecting a position along the path
    rx_pos_idx = np.random.randint(0, len(rx_positions))
    rx_pos = rx_positions[rx_pos_idx]
    rx.position = rx_pos

    state = torch.tensor(rx_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0)[None, :]

    action, log_probs_per_element = net.get_action(state)
    
    log_prob = torch.stack(log_probs_per_element).sum()

    phase_shift = action_to_phase(action)

    ris.phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, num_rows, num_cols])  # Update RIS phases

    # Compute the received power (objective) and scale by a factor for reward
    reward = compute_received_power() * 10000000

    log_prob_batch.append(log_prob)
    action_batch.append(action)
    reward_batch.append(reward)
    state_batch.append(state)

    # Update the policy every batch_size steps
    if (step + 1) % batch_size == 0:
        rewards = torch.tensor(reward_batch, dtype=torch.float32)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-6)
        loss = -torch.stack(log_prob_batch) * normalized_rewards
        loss = loss.mean()

        # Backpropagation
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
        

# Plot the mean rewards over time
plt.plot(mean_rewards)
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.show()
plt.savefig("mean_rewards.png")