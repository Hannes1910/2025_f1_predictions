#!/usr/bin/env python3
"""
Temporal Fusion Transformer with Real F1 Data
Integrates with FastF1 service and existing data infrastructure
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
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the existing data loader - we'll use FastF1 directly for now
# import sys
# sys.path.append('packages/ml-core')
# from f1_predictor.data_loader import DataLoader
import fastf1
fastf1.Cache.enable_cache('cache')  # Enable caching for faster data loading

class F1RealDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for real F1 predictions"""
    
    def __init__(self, years=[2023, 2024], batch_size=64):
        super().__init__()
        self.years = years
        self.batch_size = batch_size
        self.max_encoder_length = 5  # Use last 5 races
        self.max_prediction_length = 1  # Predict next race
        
    def prepare_data(self):
        """Load real F1 data from multiple sources"""
        print("\n📊 Loading real F1 data...")
        
        # Load from database
        conn = sqlite3.connect('f1_predictions_test.db')
        
        # Load races
        races_df = pd.read_sql_query("SELECT * FROM races ORDER BY date", conn)
        
        # Load drivers
        drivers_df = pd.read_sql_query("SELECT * FROM drivers", conn)
        
        # Create driver mappings
        driver_map = {d['code']: d['id'] for _, d in drivers_df.iterrows()}
        team_map = {d['id']: d['team'] for _, d in drivers_df.iterrows()}
        
        # Prepare time series data
        all_data = []
        
        # For 2024 season (historical data)
        for year in self.years:
            print(f"\nProcessing {year} season...")
            
            # Get qualifying and race data for each round
            for round_num in range(1, 25):  # Up to 24 races
                try:
                    # Get qualifying data
                    qual_data = self.data_loader.load_qualifying_data(year, round_num)
                    if not qual_data:
                        continue
                        
                    # Get race data
                    race_data = self.data_loader.load_race_data(year, round_num)
                    if not race_data:
                        continue
                    
                    # Get weather data
                    race_info = races_df[races_df['round'] == round_num].iloc[0]
                    coords = self.data_loader.get_circuit_coordinates()
                    circuit_coords = coords.get(race_info['circuit'], (0, 0))
                    weather = self.data_loader.get_weather_data(
                        circuit_coords[0], 
                        circuit_coords[1],
                        race_info['date']
                    )
                    
                    # Process each driver's data
                    for driver_code, qual_times in qual_data.items():
                        if driver_code not in driver_map:
                            continue
                            
                        driver_id = driver_map[driver_code]
                        team_id = team_map.get(driver_id, 1)
                        
                        # Get race result
                        race_result = race_data.get(driver_code, {})
                        
                        row = {
                            'year': year,
                            'race_round': round_num,
                            'driver_id': driver_id,
                            'driver_code': driver_code,
                            'team_id': team_id,
                            'circuit': race_info['circuit'],
                            'country': race_info['country'],
                            
                            # Qualifying data
                            'q1_time': qual_times.get('Q1', 999),
                            'q2_time': qual_times.get('Q2', 999),
                            'q3_time': qual_times.get('Q3', 999),
                            'qualifying_position': qual_times.get('position', 20),
                            
                            # Race results
                            'finishing_position': race_result.get('position', 20),
                            'points_scored': race_result.get('points', 0),
                            'fastest_lap_time': race_result.get('fastest_lap', 999),
                            'dnf': 1 if race_result.get('status', '') != 'Finished' else 0,
                            
                            # Weather data
                            'temperature': weather.get('temperature', 20),
                            'precipitation': weather.get('precipitation', 0),
                            'wind_speed': weather.get('wind_speed', 0),
                            
                            # Track characteristics
                            'track_type': self._get_track_type(race_info['circuit']),
                            'track_length': self._get_track_length(race_info['circuit']),
                            
                            # Time index for series
                            'time_idx': (year - 2023) * 24 + round_num - 1
                        }
                        
                        all_data.append(row)
                        
                except Exception as e:
                    print(f"  - Error processing round {round_num}: {e}")
                    continue
                    
                print(f"  - Processed round {round_num}")
        
        # Create DataFrame
        self.data = pd.DataFrame(all_data)
        
        # Add derived features
        self._add_derived_features()
        
        # Save processed data
        self.data.to_csv('data/f1_real_time_series.csv', index=False)
        
        print(f"\n✅ Loaded {len(self.data)} records from {len(self.data['race_round'].unique())} races")
        
        conn.close()
        
    def _get_track_type(self, circuit):
        """Categorize track types"""
        street_circuits = ['Monaco', 'Singapore', 'Las Vegas', 'Miami', 'Azerbaijan']
        hybrid_circuits = ['Canada', 'Australia', 'Mexico']
        
        if circuit in street_circuits:
            return 'street'
        elif circuit in hybrid_circuits:
            return 'hybrid'
        else:
            return 'permanent'
            
    def _get_track_length(self, circuit):
        """Get track lengths in km"""
        track_lengths = {
            'Bahrain': 5.412, 'Saudi Arabia': 6.174, 'Australia': 5.278,
            'Japan': 5.807, 'China': 5.451, 'Miami': 5.412,
            'Emilia-Romagna': 4.909, 'Monaco': 3.337, 'Canada': 4.361,
            'Spain': 4.657, 'Austria': 4.318, 'Great Britain': 5.891,
            'Hungary': 4.381, 'Belgium': 7.004, 'Netherlands': 4.259,
            'Italy': 5.793, 'Azerbaijan': 6.003, 'Singapore': 4.940,
            'United States': 5.513, 'Mexico': 4.304, 'Brazil': 4.309,
            'Las Vegas': 6.120, 'Qatar': 5.419, 'Abu Dhabi': 5.281
        }
        return track_lengths.get(circuit, 5.0)
        
    def _add_derived_features(self):
        """Add derived features for better predictions"""
        # Sort by driver and time
        self.data = self.data.sort_values(['driver_id', 'time_idx'])
        
        # Add rolling features
        for driver_id in self.data['driver_id'].unique():
            mask = self.data['driver_id'] == driver_id
            
            # Recent form (last 3 races)
            self.data.loc[mask, 'avg_position_last3'] = (
                self.data.loc[mask, 'finishing_position']
                .rolling(window=3, min_periods=1)
                .mean()
                .shift(1)
                .fillna(10)
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
            
            # Championship position (approximated)
            # This would be calculated properly with full standings
            self.data.loc[mask, 'championship_position'] = (
                self.data.loc[mask, 'cumulative_points']
                .rank(ascending=False, method='min')
            )
        
        # Convert categoricals to strings
        self.data['driver_id'] = self.data['driver_id'].astype(str)
        self.data['team_id'] = self.data['team_id'].astype(str)
        
    def setup(self, stage=None):
        """Create train/val/test datasets"""
        
        # Create dataset configuration
        self.training_cutoff = self.data['time_idx'].max() - 6  # Last 6 races for val/test
        
        # Configure TimeSeriesDataSet with real features
        self.training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= self.training_cutoff],
            time_idx="time_idx",
            target="finishing_position",
            group_ids=["driver_id"],
            
            # Static categoricals
            static_categoricals=["driver_id", "team_id"],
            
            # Time-varying known categoricals
            time_varying_known_categoricals=["track_type", "circuit"],
            
            # Time-varying known reals
            time_varying_known_reals=[
                "race_round", 
                "temperature",
                "precipitation",
                "wind_speed",
                "track_length"
            ],
            
            # Time-varying unknown reals (only known when they happen)
            time_varying_unknown_reals=[
                "qualifying_position",
                "q1_time",
                "q2_time", 
                "q3_time",
                "avg_position_last3",
                "cumulative_points",
                "dnf_rate",
                "championship_position"
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
    """Custom metric for F1 position accuracy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "position_accuracy"
        
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Calculate percentage of predictions within 3 positions"""
        # Round predictions to nearest position
        y_pred_rounded = torch.round(y_pred).clamp(1, 20)
        
        # Calculate position difference
        position_diff = torch.abs(y_pred_rounded - y_true)
        
        # Count predictions within 3 positions (more realistic for F1)
        correct = (position_diff <= 3).float()
        
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


def create_tft_model_real(training_dataset):
    """Create TFT model optimized for real F1 data"""
    
    print("\n🏗️ Creating TFT model for real F1 data...")
    
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        
        # Architecture - larger for real data complexity
        hidden_size=128,  # Increased hidden state
        lstm_layers=2,
        dropout=0.2,  # More dropout for regularization
        hidden_continuous_size=64,
        
        # Attention
        attention_head_size=8,  # More attention heads
        
        # Learning
        learning_rate=0.001,
        
        # Loss - quantile loss for uncertainty
        loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
        
        # Optimizer settings
        optimizer="adam",
        
        # Logging
        log_interval=10,
        reduce_on_plateau_patience=5,
        
        # Custom metrics
        logging_metrics=[
            F1PositionAccuracy(),
            MAE(),
            SMAPE()
        ]
    )
    
    print(f"✅ Created TFT model with {sum(p.numel() for p in tft.parameters()):,} parameters")
    
    return tft


def train_tft_real_data():
    """Train TFT on real F1 data"""
    
    print("🏎️ F1 Temporal Fusion Transformer - Real Data Training")
    print("=" * 60)
    
    # Initialize data module with real data
    data_module = F1RealDataModule(years=[2023, 2024])
    data_module.prepare_data()
    data_module.setup()
    
    # Create model
    tft = create_tft_model_real(data_module.training)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="tft-real-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            dirpath="models/tft"
        ),
        pl.callbacks.LearningRateMonitor("epoch")
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=150,  # More epochs for real data
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_checkpointing=True,
        log_every_n_steps=5,
    )
    
    # Train model
    print("\n🏋️ Training on real F1 data...")
    trainer.fit(tft, datamodule=data_module)
    
    # Test model
    print("\n🧪 Testing model...")
    test_results = trainer.test(tft, datamodule=data_module)
    
    # Save final model
    trainer.save_checkpoint("models/tft/tft_real_final.ckpt")
    
    return tft, trainer, test_results


def evaluate_predictions(model, data_module):
    """Evaluate model predictions in detail"""
    
    print("\n📊 Detailed Evaluation...")
    
    # Get predictions on test set
    test_dataloader = data_module.test_dataloader()
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            pred = model(x)
            
            # Get median prediction (index 2 for 0.5 quantile)
            if isinstance(pred, dict):
                pred = pred['prediction_outputs'][:, :, 2]
            
            predictions.append(pred)
            actuals.append(y[0])
    
    predictions = torch.cat(predictions)
    actuals = torch.cat(actuals)
    
    # Calculate metrics
    position_diff = torch.abs(predictions.round() - actuals)
    
    print(f"\nPrediction Metrics:")
    print(f"  - Mean Absolute Error: {position_diff.mean():.2f} positions")
    print(f"  - Within 1 position: {(position_diff <= 1).float().mean() * 100:.1f}%")
    print(f"  - Within 3 positions: {(position_diff <= 3).float().mean() * 100:.1f}%")
    print(f"  - Within 5 positions: {(position_diff <= 5).float().mean() * 100:.1f}%")
    

if __name__ == "__main__":
    # Train the model
    model, trainer, results = train_tft_real_data()
    
    # Evaluate in detail
    data_module = F1RealDataModule(years=[2023, 2024])
    data_module.prepare_data()
    data_module.setup()
    evaluate_predictions(model, data_module)
    
    print("\n✅ TFT Real Data Training Complete!")
    print(f"📊 Test results: {results}")
    
    # Next steps
    print("\n🚀 Next Steps:")
    print("1. Deploy model to production")
    print("2. Set up A/B testing framework")
    print("3. Monitor performance on 2025 races")