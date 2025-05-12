import requests
from collections import defaultdict

class InnosimAPI:
    def __init__(self, ip):
        self.ip = ip

    def update_association_matrix(self, mapping_array):
            tx_map = {"ris_targets": [1, 2]} | self.receiver_list_to_dict(mapping_array)
            try:
                response = requests.post(f"http://{self.ip}:5000/update-association-matrix", json=tx_map)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Error updating association matrix: {e}")

    def compute_reward(self):
        try:
            association_resp = requests.get(f"http://{self.ip}:5000/get-association-matrix")
            association_resp.raise_for_status()
            association = association_resp.json()

            tx_data = {key: value for key, value in association.items() if key.startswith('tx')}

            sim_resp = requests.post(f"http://{self.ip}:5000/run-sim")
            sim_resp.raise_for_status()
            sim_results = sim_resp.json()

            if 'output_received' not in sim_results:
                print("Error: 'output_received' missing from simulation results")
                return 0.0

            data_rates = {
                key: value for key, value in sim_results['output_received'].items() if key.startswith('tx')
            }

            summed_data_rates = {key: 0 for key in tx_data}

            for tx_key, tx_values in tx_data.items():
                for rx_key in tx_values:
                    corresponding_rx_key = f'rx{rx_key}'
                    try:
                        summed_data_rates[tx_key] += data_rates[tx_key][corresponding_rx_key]['rx_data_rate']
                    except KeyError as e:
                        print(f"Missing data for {tx_key} or {corresponding_rx_key}: {e}")
                        continue

            return sum(summed_data_rates.values())

        except requests.RequestException as e:
            print(f"Request error in compute_reward: {e}")
            return 0.0
        except ValueError as e:
            print(f"JSON decoding error in compute_reward: {e}")
            return 0.0

    def get_observation(self):
        try:
            response = requests.get(f"http://{self.ip}:5000/get-config")
            response.raise_for_status()
            data = response.json()
            return [value['position'] for key, value in data.items() if key.startswith('rx')]
        except requests.RequestException as e:
            print(f"Error getting observation: {e}")
            return []
        except (ValueError, KeyError) as e:
            print(f"Error processing observation data: {e}")
            return []

    def reset(self):
        try:
            requests.post(f"http://{self.ip}:5000/reset-config").raise_for_status()
            requests.post(f"http://{self.ip}:5000/reset-association-matrix").raise_for_status()
        except requests.RequestException as e:
            print(f"Error during reset: {e}")

    
    def receiver_list_to_dict(self, mapping_array):
        tx_map = defaultdict(list)
        for idx, tx_number in enumerate(mapping_array):
            tx_key = f"tx{tx_number + 1}"
            rx_key = idx + 1
            tx_map[tx_key].append(rx_key)
        return dict(tx_map)
    

"""
innosim = InnosimAPI(ip="3.70.154.180")
print(innosim.get_observation())

wait = input("Press Enter to continue...")
print(innosim.compute_reward())

wait = input("Press Enter to continue...")


response = requests.get(f"http://{ip}:5000/get-config")
print(response.json())

association = requests.get(f"http://{ip}:5000/get-association-matrix").json()
print(association)


transmitters = {key: value for key, value in association.items() if key.startswith('tx')}
print(transmitters)


sim_results = requests.post(f"http://{ip}:5000/run-sim").json()
results_transmitters = {key: value for key, value in sim_results['output_received'].items() if key.startswith('tx')}


print(results_transmitters)
"""