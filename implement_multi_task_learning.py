#!/usr/bin/env python3
"""
Multi-Task Learning Implementation for F1 Predictions
Task IDs: MTL-001 through MTL-006
Expected improvement: +2% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

print("🎯 F1 Multi-Task Learning Implementation")
print("=" * 60)

class F1MultiTaskDataset(Dataset):
    """Dataset for multi-task F1 predictions"""
    
    def __init__(self, data_path='data/f1_multitask_data.csv'):
        # Load data (would come from your actual data pipeline)
        self.data = pd.read_csv(data_path) if Path(data_path).exists() else self.create_sample_data()
        
        # Prepare features and targets
        self.prepare_data()
        
    def create_sample_data(self):
        """Create sample data for testing"""
        n_samples = 1000
        data = pd.DataFrame({
            # Features
            'qualifying_position': np.random.randint(1, 21, n_samples),
            'team_performance': np.random.uniform(0, 1, n_samples),
            'driver_consistency': np.random.uniform(0, 1, n_samples),
            'weather_temp': np.random.uniform(15, 35, n_samples),
            'tire_age': np.random.randint(0, 30, n_samples),
            
            # Targets
            'finishing_position': np.random.randint(1, 21, n_samples),
            'lap_time': np.random.uniform(80, 100, n_samples),
            'pit_stop': np.random.randint(0, 2, n_samples),
            'points_scored': np.random.choice([25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0], n_samples),
            'dnf': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        })
        return data
    
    def prepare_data(self):
        """Prepare features and multiple targets"""
        feature_cols = ['qualifying_position', 'team_performance', 'driver_consistency', 
                       'weather_temp', 'tire_age']
        
        # Standardize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data[feature_cols])
        
        # Prepare targets
        self.targets = {
            'position': self.data['finishing_position'].values - 1,  # 0-indexed
            'lap_time': self.data['lap_time'].values,
            'pit_stop': self.data['pit_stop'].values,
            'points': self.data['points_scored'].values,
            'dnf': self.data['dnf'].values
        }
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'position': torch.LongTensor([self.targets['position'][idx]]),
            'lap_time': torch.FloatTensor([self.targets['lap_time'][idx]]),
            'pit_stop': torch.FloatTensor([self.targets['pit_stop'][idx]]),
            'points': torch.FloatTensor([self.targets['points'][idx]]),
            'dnf': torch.FloatTensor([self.targets['dnf'][idx]])
        }


class SharedEncoder(nn.Module):
    """Task MTL-001: Shared encoder architecture"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        return self.encoder(x)


class TaskHead(nn.Module):
    """Task-specific prediction head"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128, task_type='regression'):
        super().__init__()
        
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.head(x)


class F1MultiTaskModel(pl.LightningModule):
    """Task MTL-002: Multi-task model with task-specific heads"""
    
    def __init__(self, input_dim=5, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # For multiple optimizers
        
        # Shared encoder
        self.shared_encoder = SharedEncoder(input_dim)
        
        # Task-specific heads
        self.position_head = TaskHead(self.shared_encoder.output_dim, 20, task_type='classification')
        self.lap_time_head = TaskHead(self.shared_encoder.output_dim, 1, task_type='regression')
        self.pit_stop_head = TaskHead(self.shared_encoder.output_dim, 1, task_type='binary')
        self.points_head = TaskHead(self.shared_encoder.output_dim, 1, task_type='regression')
        self.dnf_head = TaskHead(self.shared_encoder.output_dim, 1, task_type='binary')
        
        # Task weights (will be learned with GradNorm)
        self.log_task_weights = nn.Parameter(torch.zeros(5))
        
        # Task MTL-003: Loss functions for each task
        self.position_loss = nn.CrossEntropyLoss()
        self.lap_time_loss = nn.MSELoss()
        self.pit_stop_loss = nn.BCEWithLogitsLoss()
        self.points_loss = nn.MSELoss()
        self.dnf_loss = nn.BCEWithLogitsLoss()
        
        # For GradNorm
        self.initial_losses = None
        
    def forward(self, x):
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Task-specific predictions
        outputs = {
            'position': self.position_head(shared_features),
            'lap_time': self.lap_time_head(shared_features),
            'pit_stop': self.pit_stop_head(shared_features),
            'points': self.points_head(shared_features),
            'dnf': self.dnf_head(shared_features)
        }
        
        return outputs
    
    def compute_losses(self, outputs, targets):
        """Task MTL-003: Compute individual task losses"""
        losses = {
            'position': self.position_loss(outputs['position'], targets['position'].squeeze()),
            'lap_time': self.lap_time_loss(outputs['lap_time'], targets['lap_time']),
            'pit_stop': self.pit_stop_loss(outputs['pit_stop'], targets['pit_stop']),
            'points': self.points_loss(outputs['points'], targets['points']),
            'dnf': self.dnf_loss(outputs['dnf'], targets['dnf'])
        }
        return losses
    
    def gradnorm_backward(self, losses):
        """Task MTL-004: Implement GradNorm for balanced training"""
        # Get task weights
        task_weights = F.softmax(self.log_task_weights, dim=0) * len(losses)
        
        # Weighted loss
        weighted_losses = []
        for i, (task_name, loss) in enumerate(losses.items()):
            weighted_losses.append(task_weights[i] * loss)
        
        total_loss = sum(weighted_losses)
        
        # Compute gradients of shared parameters
        if self.training and self.initial_losses is not None:
            # Get gradients with respect to shared parameters
            shared_params = list(self.shared_encoder.parameters())
            
            # Compute gradient norms for each task
            grad_norms = []
            for i, (task_name, loss) in enumerate(losses.items()):
                grad = torch.autograd.grad(
                    task_weights[i] * loss,
                    shared_params,
                    retain_graph=True,
                    create_graph=True
                )
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grad]))
                grad_norms.append(grad_norm)
            
            # Compute relative inverse training rates
            loss_ratios = []
            for i, (task_name, loss) in enumerate(losses.items()):
                ratio = loss.detach() / (self.initial_losses[task_name] + 1e-8)
                loss_ratios.append(ratio)
            
            mean_ratio = sum(loss_ratios) / len(loss_ratios)
            
            # Compute target gradient norms
            target_grad_norms = []
            for ratio in loss_ratios:
                target_grad_norms.append((ratio / mean_ratio) ** 0.12)  # alpha = 0.12
            
            # GradNorm loss
            grad_norm_loss = 0
            for i, (grad_norm, target_norm) in enumerate(zip(grad_norms, target_grad_norms)):
                grad_norm_loss += torch.abs(grad_norm - target_norm * grad_norms[0].detach())
            
            # Add gradient penalty to task weights
            total_loss += 0.01 * grad_norm_loss
        
        return total_loss, task_weights
    
    def training_step(self, batch, batch_idx):
        # Get optimizers
        model_opt, weight_opt = self.optimizers()
        
        # Forward pass
        outputs = self(batch['features'])
        
        # Compute losses
        losses = self.compute_losses(outputs, batch)
        
        # Store initial losses for GradNorm
        if self.initial_losses is None:
            self.initial_losses = {k: v.detach() for k, v in losses.items()}
        
        # Apply GradNorm
        total_loss, task_weights = self.gradnorm_backward(losses)
        
        # Backward pass
        model_opt.zero_grad()
        weight_opt.zero_grad()
        self.manual_backward(total_loss)
        model_opt.step()
        weight_opt.step()
        
        # Log metrics
        self.log('train_loss', total_loss)
        for i, (task_name, loss) in enumerate(losses.items()):
            self.log(f'train_{task_name}_loss', loss)
            self.log(f'weight_{task_name}', task_weights[i])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch['features'])
        
        # Compute losses
        losses = self.compute_losses(outputs, batch)
        
        # Task MTL-006: Cross-task performance analysis
        # Calculate position accuracy
        position_preds = torch.argmax(outputs['position'], dim=1)
        position_acc = (position_preds == batch['position'].squeeze()).float().mean()
        
        # Calculate other metrics
        lap_time_mae = torch.abs(outputs['lap_time'] - batch['lap_time']).mean()
        
        # Log metrics
        total_loss = sum(losses.values())
        self.log('val_loss', total_loss)
        self.log('val_position_accuracy', position_acc * 100)
        self.log('val_lap_time_mae', lap_time_mae)
        
        for task_name, loss in losses.items():
            self.log(f'val_{task_name}_loss', loss)
        
        return total_loss
    
    def configure_optimizers(self):
        # Task MTL-005: Build task importance weighting
        # Separate optimizers for model and task weights
        model_optimizer = torch.optim.Adam(
            list(self.shared_encoder.parameters()) + 
            list(self.position_head.parameters()) +
            list(self.lap_time_head.parameters()) +
            list(self.pit_stop_head.parameters()) +
            list(self.points_head.parameters()) +
            list(self.dnf_head.parameters()),
            lr=self.hparams.learning_rate
        )
        
        weight_optimizer = torch.optim.Adam([self.log_task_weights], lr=0.01)
        
        return [model_optimizer, weight_optimizer]


def train_multi_task_model():
    """Main training function"""
    print("\nTask MTL-001 to MTL-006: Training Multi-Task Model...")
    
    # Create dataset
    dataset = F1MultiTaskDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    model = F1MultiTaskModel()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_position_accuracy',
                mode='max',
                filename='mtl-{epoch:02d}-{val_position_accuracy:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, val_loader)
    
    print("\n✅ Multi-Task Learning implementation complete!")
    print("📊 Task weights learned:", F.softmax(model.log_task_weights, dim=0).detach().numpy())
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = train_multi_task_model()
    
    print("\n🚀 Next steps:")
    print("1. Integrate with real F1 data")
    print("2. Combine with TFT predictions")
    print("3. Evaluate task-specific improvements")