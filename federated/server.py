"""
Federated learning server: aggregates client weights using FedAvg.
"""

import torch
import copy

class FLServer:
    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device
        self.global_weights = copy.deepcopy(global_model.state_dict())
        
    def aggregate_fedavg(self, client_weights):
        """
        FedAvg: weighted average of client weights.
        """
        if not client_weights:
            return
        
        # Initialize with first client's weights
        avg_weights = copy.deepcopy(client_weights[0])
        
        # Sum all weights
        for key in avg_weights.keys():
            for w in client_weights[1:]:
                avg_weights[key] += w[key]
            
            # Average
            avg_weights[key] = avg_weights[key] / len(client_weights)
        
        # Update global model
        self.global_model.load_state_dict(avg_weights)
        self.global_weights = avg_weights
        
        return avg_weights
    
    def get_weights(self):
        """Get current global model weights."""
        return copy.deepcopy(self.global_weights)
    
    def set_weights(self, weights):
        """Set global model weights."""
        self.global_model.load_state_dict(weights)
        self.global_weights = copy.deepcopy(weights)