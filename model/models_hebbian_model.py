import torch
import torch.nn as nn

class HebbianMemory(nn.Module):
    """
    Implements the BDH mechanism: synaptic state σ updated via Hebbian rule.
    
    From BDH paper: σ_t = σ_{t-1} + η * (y_{t-1} ⊗ x_t)
    where ⊗ is outer product.
    """
    def __init__(self, n_neurons=128, state_dim=64, lr=0.01):
        super().__init__()
        self.n = n_neurons
        self.d = state_dim
        self.lr = lr
        
        # The synaptic state matrix (this is what we visualize)
        # In BDH, this is the "fast weight" memory
        self.sigma = nn.Parameter(
            torch.zeros(n_neurons, state_dim), 
            requires_grad=False  # Hebbian update, not gradient descent
        )
        
        # Encoder/decoder (like BDH's E and D matrices)
        self.encoder = nn.Linear(state_dim, n_neurons)
        self.decoder = nn.Linear(state_dim, state_dim)
        
        # Track activation sparsity (BDH reports ~5% activity)
        self.activation_threshold = 0.1
        
    def hebbian_update(self, x, y):
        """
        The core BDH mechanism:
        When x (pre-synaptic) and y (post-synaptic) fire together,
        strengthen the connection σ.
        """
        # Outer product: x [batch, n] * y [batch, d] -> [batch, n, d]
        outer = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
        
        # Update σ with learning rate (Hebbian plasticity)
        self.sigma.data += self.lr * outer.mean(dim=0)
        
        # Optional: Decay (like BDH's damping parameter u)
        self.sigma.data *= 0.99
        
    def forward(self, query, store_pairs=None):
        """
        query: [batch, d] - what we want to recall
        store_pairs: optional (key, value) to store first
        """
        batch_size = query.shape[0]
        
        # If storing new associations (write phase)
        if store_pairs is not None:
            key, value = store_pairs
            encoded_key = torch.relu(self.encoder(key))
            self.hebbian_update(encoded_key, value)
        
        # Read phase: query @ σ
        encoded_query = torch.relu(self.encoder(query))
        
        # Apply sparsity (BDH uses sparse positive activations)
        mask = (encoded_query > self.activation_threshold).float()
        sparse_query = encoded_query * mask
        
        # Retrieve from synaptic memory
        retrieved = sparse_query @ self.sigma  # [batch, d]
        
        # Decode to output space
        output = self.decoder(retrieved)
        
        return {
            'output': output,
            'sigma': self.sigma.clone(),  # For visualization
            'sparsity': (sparse_query > 0).float().mean()  # Track activity level
        }
    
    def reset_memory(self):
        """Clear the synaptic state (new episode)"""
        self.sigma.data.zero_()