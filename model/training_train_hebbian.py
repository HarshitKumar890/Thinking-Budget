import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
from pathlib import Path

from models_hebbian_model import HebbianMemory
from data_generator import AssociativeTask

def train():
    # Load config
    with open('configs_hebbian_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = HebbianMemory(
        n_neurons=config['model']['n_neurons'],
        state_dim=config['model']['state_dim'],
        lr=config['model']['learning_rate']
    ).to(device)
    
    # Only encoder/decoder need gradients (σ is Hebbian)
    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=0.001)
    
    dataset = AssociativeTask(num_pairs=8, embed_dim=config['model']['state_dim'])
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    losses = []
    
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for batch in loader:
            query = batch['query'].to(device)
            target = batch['target'].to(device)
            
            # Reset memory each episode (like BDH processing new context)
            model.reset_memory()
            
            # Store associations first (write phase)
            # In real training, we'd show multiple pairs then query one
            with torch.no_grad():
                # Simulate storing 8 pairs quickly
                for i in range(8):
                    fake_key = torch.randn_like(query)
                    fake_value = torch.randn_like(target)
                    model(query, store_pairs=(fake_key, fake_value))
            
            # Now query
            result = model(query)
            output = result['output']
            
            # Loss: MSE between retrieved and target
            loss = nn.MSELoss()(output, target)
            
            # Backprop only through encoder/decoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Sparsity: {result['sparsity']:.2%}")
    
    # Save for web
    Path('export/models').mkdir(parents=True, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), 'export/models/hebbian_weights.pth')
    
    # Save loss curve
    with open('export/models/hebbian_training.json', 'w') as f:
        json.dump({'losses': losses}, f)
    
    # Save sample state matrix for visualization
    sample_sigma = result['sigma'].cpu().numpy().tolist()
    with open('export/models/hebbian_sigma.json', 'w') as f:
        json.dump({'sigma': sample_sigma}, f)
    
    print("Training complete. Weights exported to export/models/")

if __name__ == '__main__':
    train()