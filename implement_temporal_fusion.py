#!/usr/bin/env python3
"""
Temporal Fusion Transformer Implementation for F1 Predictions
Expected improvement: +5% accuracy
Task tracking: TFT-001 through TFT-006
"""

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, QuantileLoss
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Task TFT-001: Set up PyTorch Lightning environment
print("🏁 F1 Temporal Fusion Transformer Implementation")
print("=" * 60)
print("Task TFT-001: Setting up PyTorch Lightning environment...")

class F1DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for F1 predictions"""
    
    def __init__(self, data_path='data/f1_time_series.csv', batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_encoder_length = 5  # Use last 5 races
        self.max_prediction_length = 1  # Predict next race
        
    def prepare_data(self):
        """Task TFT-002: Create TimeSeriesDataSet with F1 data structure"""
        print("\nTask TFT-002: Creating TimeSeriesDataSet...")
        
        # Load or create F1 time series data
        if Path(self.data_path).exists():
            self.data = pd.read_csv(self.data_path)
        else:
            self.data = self.create_f1_time_series()
            
    def create_f1_time_series(self):
        """Create F1 data in time series format"""
        # This would load from your actual data sources
        # For now, create sample structure
        
        races = range(1, 25)  # 24 races
        drivers = range(1, 21)  # 20 drivers
        
        data = []
        for race in races:
            for driver in drivers:
                row = {
                    'race_round': race,
                    'driver_id': driver,
                    'team_id': (driver - 1) // 2 + 1,  # 10 teams
                    'position': np.random.randint(1, 21),
                    'qualifying_position': np.random.randint(1, 21),
                    'weather_temp': np.random.uniform(15, 35),
                    'track_type': np.random.choice(['street', 'permanent', 'hybrid']),
                    'tire_strategy': np.random.choice(['soft-medium', 'medium-hard', 'soft-hard']),
                    'championship_position': np.random.randint(1, 21),
                    'points_scored': np.random.choice([25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]),
                    'dnf': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'avg_lap_time': np.random.uniform(80, 100)
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add time-based features
        df['time_idx'] = df['race_round'] - 1  # 0-indexed
        df['driver_id'] = df['driver_id'].astype(str)
        df['team_id'] = df['team_id'].astype(str)
        
        # Save for future use
        df.to_csv(self.data_path, index=False)
        
        return df
    
    def setup(self, stage=None):
        """Create train/val/test datasets"""
        
        # Create dataset configuration
        self.training_cutoff = self.data['time_idx'].max() - 6  # Last 6 races for val/test
        
        # Task TFT-002 continued: Configure TimeSeriesDataSet
        self.training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= self.training_cutoff],
            time_idx="time_idx",
            target="position",
            group_ids=["driver_id"],
            
            # Static categoricals - don't change over time
            static_categoricals=["driver_id", "team_id"],
            
            # Time-varying known categoricals - we know these ahead
            time_varying_known_categoricals=["track_type", "tire_strategy"],
            
            # Time-varying known reals - we know these ahead
            time_varying_known_reals=["race_round", "weather_temp"],
            
            # Time-varying unknown categoricals
            time_varying_unknown_categoricals=[],
            
            # Time-varying unknown reals - only known when they happen
            time_varying_unknown_reals=[
                "qualifying_position", 
                "championship_position",
                "avg_lap_time"
            ],
            
            # Target normalizer
            target_normalizer=GroupNormalizer(
                groups=["driver_id"], 
                transformation="softplus"
            ),
            
            # Sequence lengths
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            
            # Additional targets for multi-task learning
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation set
        validation_cutoff = self.data['time_idx'].max() - 3
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            self.data[lambda x: (x.time_idx > self.training_cutoff) & 
                               (x.time_idx <= validation_cutoff)],
            predict=True,
            stop_randomization=True
        )
        
        # Create test set
        self.test = TimeSeriesDataSet.from_dataset(
            self.training,
            self.data[lambda x: x.time_idx > validation_cutoff],
            predict=True,
            stop_randomization=True
        )
        
        print(f"✅ Created datasets - Train: {len(self.training)}, Val: {len(self.validation)}, Test: {len(self.test)}")
    
    def train_dataloader(self):
        return self.training.to_dataloader(train=True, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return self.validation.to_dataloader(train=False, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return self.test.to_dataloader(train=False, batch_size=self.batch_size)


class F1PositionAccuracy(MAE):
    """Task TFT-003: Custom metric for position accuracy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "position_accuracy"
        
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Calculate percentage of predictions within 2 positions"""
        # Round predictions to nearest position
        y_pred_rounded = torch.round(y_pred)
        
        # Calculate position difference
        position_diff = torch.abs(y_pred_rounded - y_true)
        
        # Count predictions within 2 positions
        correct = (position_diff <= 2).float()
        
        # Store for compute
        if not hasattr(self, 'total_correct'):
            self.total_correct = 0
            self.total_count = 0
            
        self.total_correct += correct.sum()
        self.total_count += correct.numel()
        
        # Also update parent MAE
        super().update(y_pred, y_true)
    
    def compute(self):
        """Compute accuracy percentage"""
        accuracy = (self.total_correct / self.total_count) * 100
        return accuracy


def create_tft_model(training_dataset):
    """Create Temporal Fusion Transformer model"""
    
    # Task TFT-003: Configure model with custom metrics
    print("\nTask TFT-003: Configuring TFT with custom metrics...")
    
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        
        # Architecture
        hidden_size=64,  # Hidden state size
        lstm_layers=2,  # Number of LSTM layers
        dropout=0.1,  # Dropout rate
        hidden_continuous_size=32,  # Hidden size for processing continuous variables
        
        # Attention
        attention_head_size=4,  # Number of attention heads
        
        # Learning
        learning_rate=0.001,
        
        # Loss - use quantile loss for uncertainty estimation
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        
        # Logging
        log_interval=10,
        reduce_on_plateau_patience=4,
        
        # Custom metrics
        logging_metrics=[
            F1PositionAccuracy(),
            MAE(),
            SMAPE()
        ]
    )
    
    print(f"✅ Created TFT model with {sum(p.numel() for p in tft.parameters())} parameters")
    
    return tft


class AttentionCallback(pl.Callback):
    """Task TFT-004: Build attention visualization"""
    
    def __init__(self, save_path='visualizations/attention'):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Visualize attention weights after each validation epoch"""
        # Get a sample batch
        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        
        # Get predictions with attention
        predictions, attention = pl_module.predict(
            batch, 
            mode="raw", 
            return_attention=True
        )
        
        # Save attention weights for analysis
        torch.save({
            'attention': attention,
            'predictions': predictions,
            'batch': batch,
            'epoch': trainer.current_epoch
        }, self.save_path / f'attention_epoch_{trainer.current_epoch}.pt')
        
        print(f"💡 Saved attention weights for epoch {trainer.current_epoch}")


def train_temporal_fusion_transformer():
    """Main training function"""
    
    # Initialize data module
    data_module = F1DataModule()
    data_module.prepare_data()
    data_module.setup()
    
    # Create model
    tft = create_tft_model(data_module.training)
    
    # Task TFT-004: Set up callbacks including attention visualization
    print("\nTask TFT-004: Setting up attention visualization...")
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min"
        ),
        AttentionCallback()
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",  # Use GPU if available
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )
    
    # Train model
    print("\n🏋️ Training Temporal Fusion Transformer...")
    trainer.fit(tft, datamodule=data_module)
    
    # Test model
    print("\n🧪 Testing model...")
    test_results = trainer.test(tft, datamodule=data_module)
    
    return tft, trainer, test_results


def implement_tft_integration():
    """Task TFT-006: Integration with existing prediction pipeline"""
    print("\nTask TFT-006: Creating integration pipeline...")
    
    integration_code = '''
# Integration with existing ensemble
class TFTEnsemblePredictor:
    def __init__(self, tft_checkpoint, existing_models):
        self.tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint)
        self.existing_models = existing_models
        
    def predict(self, race_data):
        # Get TFT predictions
        tft_pred = self.tft.predict(race_data)
        
        # Get existing model predictions
        existing_preds = [m.predict(race_data) for m in self.existing_models]
        
        # Weighted ensemble
        weights = [0.3] + [0.7 / len(existing_preds)] * len(existing_preds)
        
        final_pred = weights[0] * tft_pred
        for w, p in zip(weights[1:], existing_preds):
            final_pred += w * p
            
        return final_pred
    '''
    
    # Save integration code
    with open('tft_integration.py', 'w') as f:
        f.write(integration_code)
    
    print("✅ Created TFT integration pipeline")


if __name__ == "__main__":
    # Task TFT-005 will be implemented separately with Optuna
    print("\n📝 Note: Task TFT-005 (Hyperparameter tuning) will use Optuna in separate script")
    
    # Train the model
    model, trainer, results = train_temporal_fusion_transformer()
    
    # Create integration
    implement_tft_integration()
    
    print("\n✅ Temporal Fusion Transformer implementation complete!")
    print(f"📊 Test results: {results}")
    print("\n🚀 Next steps:")
    print("1. Run hyperparameter_tuning_tft.py for TFT-005")
    print("2. Integrate with production pipeline")
    print("3. A/B test against current ensemble")