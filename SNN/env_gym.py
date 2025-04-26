import sionna as sn
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
import tensorflow as tf
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

set_random_seed(seed = 42)

# The environment class for the RIS simulation
# Gymnasium based version to allow for training with Stable Baselines
# This class is responsible for setting up the scene, generating receiver positions,
# computing channel gains, and calculating rewards based on the received power.


class RISGymEnvironment(gym.Env):
    def __init__(self, num_receivers, ris_dims = [3, 3], abs_receiver_position_bounds = [10, 10], receiver_height = 5, options_per_element = 4):
        super().__init__()

        self.num_receivers = num_receivers
        self.abs_receiver_position_bounds = abs_receiver_position_bounds # [70, 45] is maximum, derived from scene
        self.receiver_height = receiver_height

        self.evaluation_positions = []
        self.evaluation_using_ris = []

        for i in range(100):
            self.evaluation_positions.append(self.generate_rx_positions())
            self.evaluation_using_ris.append([1] + [random.choice([0, 1]) for _ in range(self.num_receivers - 1)])

        # Sionna set up

        self.scene = sn.rt.load_scene(sn.rt.scene.simple_street_canyon)
        self.scene.tx_array = sn.rt.PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
        self.scene.rx_array = sn.rt.PlanarArray(num_rows=8,
                          num_cols=8,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")

        camera = sn.rt.Camera("Cam", [0, 0, 300], orientation=[np.pi/2, np.pi/2, 0])
        self.scene.add(camera)

        tx = sn.rt.Transmitter(name="tx", position=[-32,10,32])
        self.scene.add(tx)

        self.rx_positions = self.generate_rx_positions()
        for i in range(self.num_receivers):
            rx = sn.rt.Receiver(name=f"rx{i}", position=self.rx_positions[i])
            self.scene.add(rx)

        self.rx_using_ris = [1] + [random.choice([0, 1]) for _ in range(self.num_receivers - 1)]
        random.shuffle(self.rx_using_ris)


        self.ris_position = [32, -9, 32]
        self.ris_look_at = [-5, 30, 17]
        self.ris_num_rows = ris_dims[0]
        self.ris_num_cols = ris_dims[1]
        
        cell_grid = sn.rt.CellGrid(self.ris_num_rows, self.ris_num_cols)
        ris = sn.rt.RIS(name="ris", position=self.ris_position, num_rows=self.ris_num_rows, num_cols=self.ris_num_cols, look_at=self.ris_look_at)
        ris.phase_profile = sn.rt.DiscretePhaseProfile(cell_grid=cell_grid, num_modes=1, values=tf.zeros
                                                            ([1, self.ris_num_rows, self.ris_num_cols]))
        
        self.scene.add(ris)
        self.scene.frequency = 2.4e9


        # Gymnasium environment setup

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(ris_dims[0] * ris_dims[1],), dtype=np.float32)

        rx_position_low = [-abs_receiver_position_bounds[0], -abs_receiver_position_bounds[1]] * num_receivers
        rx_position_high = [abs_receiver_position_bounds[0], abs_receiver_position_bounds[1]] * num_receivers
        ris_flags_low = [0] * num_receivers
        ris_flags_high = [1] * num_receivers

        low = np.array(rx_position_low + ris_flags_low, dtype=np.float32)
        high = np.array(rx_position_high + ris_flags_high, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_step = 0
        self.reward_history = []
        self.running_reward = 0.0
        self.data_rate_history = []
        self.running_data_rate = 0.0


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.rx_positions = self.generate_rx_positions()
        self.update_rx_positions()

        # Define which RXs use the RIS, e.g. randomly or all True
        self.rx_using_ris = [1] + [random.choice([0, 1]) for _ in range(self.num_receivers - 1)]
        random.shuffle(self.rx_using_ris)

        rx_positions_flat = np.array(self.rx_positions, dtype=np.float32).flatten()
        rx_flags = np.array(self.rx_using_ris, dtype=np.float32)

        obs = np.concatenate((rx_positions_flat, rx_flags))

        return obs, {}
    
    def step(self, action):

        self.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(action), [1, self.ris_num_rows, self.ris_num_cols])
        _, reward_real = self.calculate_reward()


       # source = self.scene.get("tx").position
       # target = self.scene.get("rx0").position


       # self.scene.get("ris").phase_gradient_reflector(source, target)
       # _, reward_phase_grad = self.calculate_reward()

        #self.scene.get("ris").focusing_lens(source, target)
       # _, reward_lens = self.calculate_reward()

        #print(reward_real)
        #print(reward_phase_grad)
        #print(reward_lens)

        
        self.scene.get("ris").phase_profile.values = tf.zeros([1, self.ris_num_rows, self.ris_num_cols])
        _, reward_baseline = self.calculate_reward()


        reward = (sum(reward_real) - sum(reward_baseline)) * 1000

        # reward = sum(reward_real) * 1000

        self.running_reward += reward
        self.current_step += 1

        # Logging and evaluation    
        if self.current_step % 200 == 0:
            self.reward_history.append(self.running_reward / 200)
            self.running_reward = 0.0
                    
        # Prepare next observation
        self.rx_positions = self.generate_rx_positions()
        self.update_rx_positions()

        self.rx_using_ris = [1] + [random.choice([0, 1]) for _ in range(self.num_receivers - 1)]
        random.shuffle(self.rx_using_ris)

        rx_positions_flat = np.array(self.rx_positions, dtype=np.float32).flatten()
        rx_flags = np.array(self.rx_using_ris, dtype=np.float32)

        observation = np.concatenate((rx_positions_flat, rx_flags))
        
        return observation, reward, False, False, {}
    
    def evaluate(self, model):

        for i in range(len(self.evaluation_positions)):
            self.rx_positions = self.evaluation_positions[i]
            self.update_rx_positions()
            self.rx_using_ris = self.evaluation_using_ris[i]
            action, _ = model.predict(np.concatenate((np.array(self.rx_positions).flatten(), np.array(self.rx_using_ris))), deterministic=True)
            self.scene.get("ris").phase_profile.values = tf.reshape(tf.convert_to_tensor(action), [1, self.ris_num_rows, self.ris_num_cols])
            _, data_rate = self.calculate_reward()
            self.running_data_rate += sum(data_rate)
        self.data_rate_history.append(self.running_data_rate/len(self.evaluation_positions))
        print(f"({len(self.data_rate_history)}) Average data rate: {self.running_data_rate/len(self.evaluation_positions)}")
        self.running_data_rate = 0.0


    def position_is_blocked(self, x, y):

        # Define invalid positions for receivers
        # Buildings, space behind buildings etc.
        # (x1, y1, x2, y2)
        # (x1, y1) - bottom-left 
        # (x2, y2) - top-right
        invalid_positions = [
            (28, 50, 80, 5),
            (28, -5, 80, -50),
            (-20, 50, 20, 5),
            (-20, -5, 20, -50),
            (-80, 50, -28, 5),
            (-80, -5, -28, -50)
        ]

        for (x1, y1, x2, y2) in invalid_positions:
            if x1 <= x <= x2 and y2 <= y <= y1:
                return True
        return False


    def generate_rx_positions(self):
        max_abs_x = self.abs_receiver_position_bounds[0]
        max_abs_y = self.abs_receiver_position_bounds[1]

        rx_positions = []

        for i in range(self.num_receivers):
            random_x = random.randint(-max_abs_x, max_abs_x)
            random_y = random.randint(-max_abs_y, max_abs_y)
            while self.position_is_blocked(random_x, random_y):
                random_x = random.randint(-max_abs_x, max_abs_x)
                random_y = random.randint(-max_abs_y, max_abs_y)
            rx_positions.append([random_x, random_y])
        
        return rx_positions
    

    def remove_nans(self, tensor):
        if tensor.dtype in (tf.complex64, tf.complex128):
            # Complex tensor handling
            real_part = tf.math.real(tensor)
            imag_part = tf.math.imag(tensor)
            is_nan = tf.math.logical_or(
                tf.math.is_nan(real_part),
                tf.math.is_nan(imag_part)
            )
        else:
            # Float tensor handling
            is_nan = tf.math.is_nan(tensor)
        
        # Create zeros with the same type as input tensor
        zeros = tf.zeros_like(tensor)
        
        # Replace NaN values with zeros
        cleaned_tensor = tf.where(is_nan, zeros, tensor)

        return cleaned_tensor


    def update_rx_positions(self):
        for i in range(self.num_receivers):
            self.scene.get(f"rx{i}").position = self.rx_positions[i] + [self.receiver_height]


    def compute_channel_gains(self, a, tau, ris=False):
        fft_size = 1
        subcarrier_spacing = 15e3
        channels = []
        
        # Define OFDM parameters
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

        # Compute the frequency-domain channel
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)

        for i in range(self.num_receivers):
            if self.rx_using_ris[i] == ris:
                h_freq_i = h_freq[:, i, :, :, :, :, :]
                channel_i = tf.reduce_mean(tf.abs(h_freq_i) ** 2)
                channels.append(channel_i)

        return channels    


    def compute_data_rates(self, channels):
        epsilon = 1e-11
        total_data_rates = []
        noise_power = 1e-7
        for i in range(len(channels)):
            desired = tf.maximum(channels[i], epsilon)
            sinr = desired / (noise_power + epsilon)
            total_data_rate = tf.math.log(1 + sinr) / tf.math.log(2.0)
            total_data_rates.append(total_data_rate.numpy())
        return total_data_rates


    # Utility function to compute received power
    def calculate_reward(self):
        
        paths = self.scene.compute_paths(max_depth=2, los=True, reflection=True, ris=True)

        # sn.rt.Scene.render_to_file(self=self.scene, camera="Cam", paths=paths, filename="scene.png")

        # Only paths that are from LOS and reflection
        a_no_ris, tau_no_ris = paths.cir(los=True, reflection=True, ris=False, num_paths=3)


        # Only paths that are from reflection and RIS
        a_ris_reflection, tau_ris_reflection = paths.cir(los=False, reflection=True, ris=True, num_paths=3)
        
        # Replace nan values with zeros. Only necessary for RIS
        # Nan values do not appear in other paths. Maybe Sionna bug?
        a_ris_reflection = self.remove_nans(a_ris_reflection)
        tau_ris_reflection = self.remove_nans(tau_ris_reflection)

        # Only paths that are from reflection
        a_only_reflection, tau_only_reflection = paths.cir(los=False, reflection=True, ris=False, num_paths=3)


        # We "subtract" reflection only paths from RIS + reflection paths
        # to get paths that are either directly from RIS or from RIS
        # and then reflected.
        # Necessary because RIS + reflection paths include the reflection
        # only paths and RIS only paths do not include reflections.

        # Mask to check whether two paths are the same.
        # We leave tau the same, as if a is zeroed, the delay
        # is irrelevant.
        mask = tf.math.reduce_all(tf.abs(a_ris_reflection - a_only_reflection) < 1e-6, axis=-1)
        mask = tf.expand_dims(mask, axis=-1) 
        a_ris = tf.where(mask, tf.zeros_like(a_ris_reflection), a_ris_reflection)
        tau_ris = tau_ris_reflection

        
        channels = self.compute_channel_gains(a_no_ris, tau_no_ris, ris=False)
        channels_ris = self.compute_channel_gains(a_ris, tau_ris, ris=True)


        total_data_rates = self.compute_data_rates(channels)
        total_data_rates_ris = self.compute_data_rates(channels_ris)

        return total_data_rates, total_data_rates_ris