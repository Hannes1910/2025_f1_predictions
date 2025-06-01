#!/usr/bin/env python3
"""
Temporal Fusion Transformer with Real F1 Data - Simplified Version
Uses FastF1 to load real race data
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
import json
import warnings
warnings.filterwarnings('ignore')

import fastf1
fastf1.Cache.enable_cache('cache')  # Enable caching


class F1RealDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for real F1 predictions"""
    
    def __init__(self, years=[2023, 2024], batch_size=64):
        super().__init__()
        self.years = years
        self.batch_size = batch_size
        self.max_encoder_length = 5  # Use last 5 races
        self.max_prediction_length = 1  # Predict next race
        
    def prepare_data(self):
        """Load real F1 data using FastF1"""
        print("\n📊 Loading real F1 data from FastF1...")
        
        all_data = []
        
        for year in self.years:
            print(f"\nProcessing {year} season...")
            
            # Get the schedule for the year
            try:
                schedule = fastf1.get_event_schedule(year)
                races = schedule[schedule['EventFormat'] != 'testing']
                
                for idx, race in races.iterrows():
                    round_num = race['RoundNumber']
                    if round_num == 0:
                        continue
                        
                    print(f"  Loading Round {round_num}: {race['EventName']}...")
                    
                    try:
                        # Load session data
                        session = fastf1.get_session(year, round_num, 'R')
                        session.load()
                        
                        # Get results
                        results = session.results
                        
                        # Load qualifying for starting positions
                        quali_session = fastf1.get_session(year, round_num, 'Q')
                        quali_session.load()
                        quali_results = quali_session.results
                        
                        # Process each driver
                        for _, driver_result in results.iterrows():
                            driver_code = driver_result['Abbreviation']
                            
                            # Get qualifying position
                            quali_pos = 20  # Default
                            if driver_code in quali_results['Abbreviation'].values:
                                quali_row = quali_results[quali_results['Abbreviation'] == driver_code].iloc[0]
                                quali_pos = quali_row['Position'] if not pd.isna(quali_row['Position']) else 20
                            
                            row = {
                                'year': year,
                                'race_round': round_num,
                                'time_idx': (year - min(self.years)) * 24 + round_num - 1,
                                'driver_code': driver_code,
                                'driver_id': str(hash(driver_code) % 1000),  # Simple hash for ID
                                'team_id': str(hash(driver_result['TeamName']) % 100),
                                'team_name': driver_result['TeamName'],
                                
                                # Race data
                                'finishing_position': float(driver_result['Position']) if not pd.isna(driver_result['Position']) else 20.0,
                                'grid_position': float(driver_result['GridPosition']) if not pd.isna(driver_result['GridPosition']) else 20.0,
                                'qualifying_position': float(quali_pos),
                                'points_scored': float(driver_result['Points']) if not pd.isna(driver_result['Points']) else 0.0,
                                'status': driver_result['Status'],
                                'dnf': 0.0 if driver_result['Status'] == 'Finished' else 1.0,
                                
                                # Track info
                                'circuit': race['Location'],
                                'country': race['Country'],
                                'track_type': self._get_track_type(race['Location']),
                                
                                # Weather placeholder (would need weather API)
                                'temperature': 25.0,
                                'precipitation': 0.0,
                                'wind_speed': 10.0,
                            }
                            
                            all_data.append(row)
                            
                    except Exception as e:
                        print(f"    Error loading race data: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Error loading {year} schedule: {e}")
                continue
        
        # Create DataFrame
        self.data = pd.DataFrame(all_data)
        
        if len(self.data) == 0:
            print("⚠️ No data loaded, creating synthetic data for testing...")
            self.data = self._create_synthetic_data()
        else:
            # Add derived features
            self._add_derived_features()
            print(f"\n✅ Loaded {len(self.data)} records from {len(self.data['race_round'].unique())} races")
        
        # Save processed data
        Path('data').mkdir(exist_ok=True)
        self.data.to_csv('data/f1_tft_data.csv', index=False)
        
    def _create_synthetic_data(self):
        """Create synthetic data for testing if real data fails"""
        data = []
        
        # Create data for 20 drivers over 20 races
        for race in range(1, 21):
            for driver in range(1, 21):
                row = {
                    'year': 2024,
                    'race_round': race,
                    'time_idx': race - 1,
                    'driver_code': f'DR{driver}',
                    'driver_id': str(driver),
                    'team_id': str((driver - 1) // 2 + 1),
                    'team_name': f'Team{(driver - 1) // 2 + 1}',
                    'finishing_position': float(np.random.randint(1, 21)),
                    'grid_position': float(np.random.randint(1, 21)),
                    'qualifying_position': float(np.random.randint(1, 21)),
                    'points_scored': float(np.random.choice([25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0])),
                    'status': 'Finished',
                    'dnf': 0.0,
                    'circuit': f'Circuit{race}',
                    'country': f'Country{race}',
                    'track_type': np.random.choice(['street', 'permanent', 'hybrid']),
                    'temperature': float(np.random.uniform(15, 35)),
                    'precipitation': 0.0,
                    'wind_speed': float(np.random.uniform(5, 20)),
                }
                data.append(row)
                
        return pd.DataFrame(data)
        
    def _get_track_type(self, location):
        """Categorize track types"""
        street_circuits = ['Monaco', 'Singapore', 'Las Vegas', 'Miami', 'Baku']
        if any(street in location for street in street_circuits):
            return 'street'
        else:
            return 'permanent'
            
    def _add_derived_features(self):
        """Add derived features for better predictions"""
        # Sort by driver and time
        self.data = self.data.sort_values(['driver_id', 'time_idx'])
        
        # Add rolling features per driver
        for driver_id in self.data['driver_id'].unique():
            mask = self.data['driver_id'] == driver_id
            
            # Recent form (last 3 races)
            self.data.loc[mask, 'avg_position_last3'] = (
                self.data.loc[mask, 'finishing_position']
                .rolling(window=3, min_periods=1)
                .mean()
                .shift(1)
                .fillna(10.5)
            )
            
            # Points momentum
            self.data.loc[mask, 'cumulative_points'] = (
                self.data.loc[mask, 'points_scored']
                .cumsum()
                .shift(1)
                .fillna(0)
            )
            
            # DNF rate
            self.data.loc[mask, 'dnf_rate'] = (
                self.data.loc[mask, 'dnf']
                .rolling(window=5, min_periods=1)
                .mean()
                .shift(1)
                .fillna(0.1)
            )
        
        # Ensure all columns are numeric where needed
        numeric_cols = ['finishing_position', 'qualifying_position', 'grid_position', 
                       'points_scored', 'temperature', 'precipitation', 'wind_speed',
                       'avg_position_last3', 'cumulative_points', 'dnf_rate']
        
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
    def setup(self, stage=None):
        """Create train/val/test datasets"""
        
        # Create dataset configuration
        self.training_cutoff = self.data['time_idx'].max() - 4  # Last 4 races for val/test
        
        # Configure TimeSeriesDataSet
        self.training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= self.training_cutoff],
            time_idx="time_idx",
            target="finishing_position",
            group_ids=["driver_id"],
            
            # Static categoricals
            static_categoricals=["driver_id", "team_id"],
            
            # Time-varying known categoricals
            time_varying_known_categoricals=["track_type"],
            
            # Time-varying known reals
            time_varying_known_reals=[
                "race_round", 
                "temperature",
                "precipitation",
                "wind_speed"
            ],
            
            # Time-varying unknown reals
            time_varying_unknown_reals=[
                "qualifying_position",
                "grid_position",
                "avg_position_last3",
                "cumulative_points",
                "dnf_rate"
            ],
            
            # Target normalizer
            target_normalizer=GroupNormalizer(
                groups=["driver_id"], 
                transformation="softplus"
            ),
            
            # Sequence lengths
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            
            # Additional settings
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation set
        validation_cutoff = self.data['time_idx'].max() - 2
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
        return self.training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
    
    def val_dataloader(self):
        return self.validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return self.test.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)


class F1PositionAccuracy(MAE):
    """Custom metric for F1 position accuracy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "position_accuracy"
        self.total_correct = 0
        self.total_count = 0
        
    def update(self, y_pred: torch.Tensor, target: torch.Tensor):
        """Calculate percentage of predictions within 3 positions"""
        # Extract actual values from target tuple
        if isinstance(target, tuple):
            y_true = target[0]
        else:
            y_true = target
            
        # Handle quantile predictions
        if len(y_pred.shape) == 3:
            # Take the median prediction (middle quantile)
            y_pred = y_pred[:, :, y_pred.shape[2] // 2]
        
        # Flatten tensors
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        
        # Round predictions to nearest position
        y_pred_rounded = torch.round(y_pred).clamp(1, 20)
        
        # Calculate position difference
        position_diff = torch.abs(y_pred_rounded - y_true)
        
        # Count predictions within 3 positions
        correct = (position_diff <= 3).float()
        
        self.total_correct += correct.sum().item()
        self.total_count += correct.numel()
        
        # Also update parent MAE
        super().update(y_pred, (y_true,))
    
    def compute(self):
        """Compute accuracy percentage"""
        if self.total_count == 0:
            return torch.tensor(0.0)
        accuracy = (self.total_correct / self.total_count) * 100
        return torch.tensor(accuracy)


def create_tft_model(training_dataset):
    """Create TFT model"""
    
    print("\n🏗️ Creating TFT model...")
    
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        
        # Architecture
        hidden_size=64,
        lstm_layers=2,
        dropout=0.1,
        hidden_continuous_size=32,
        
        # Attention
        attention_head_size=4,
        
        # Learning
        learning_rate=0.003,
        
        # Loss - quantile loss for uncertainty
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
    
    print(f"✅ Created TFT model with {sum(p.numel() for p in tft.parameters()):,} parameters")
    
    return tft


def train_tft():
    """Main training function"""
    
    print("🏎️ F1 Temporal Fusion Transformer - Real Data Training")
    print("=" * 60)
    
    # Initialize data module
    data_module = F1RealDataModule(years=[2023, 2024])
    data_module.prepare_data()
    data_module.setup()
    
    # Create model
    tft = create_tft_model(data_module.training)
    
    # Set up callbacks
    Path('models/tft').mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            dirpath="models/tft"
        ),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=50,  # Start with fewer epochs
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )
    
    # Train model
    print("\n🏋️ Training TFT on F1 data...")
    trainer.fit(tft, datamodule=data_module)
    
    # Test model
    print("\n🧪 Testing model...")
    test_results = trainer.test(tft, datamodule=data_module)
    
    return tft, trainer, test_results


if __name__ == "__main__":
    # Train the model
    model, trainer, results = train_tft()
    
    print("\n✅ TFT Training Complete!")
    print(f"📊 Test results: {results}")
    
    # Save model info
    with open('models/tft/results.json', 'w') as f:
        json.dump({
            'test_results': results,
            'model_type': 'temporal_fusion_transformer',
            'training_years': [2023, 2024]
        }, f, indent=2)
    
    print("\n🚀 Model saved to models/tft/")