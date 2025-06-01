#!/usr/bin/env python3
"""
Task TFT-005: Hyperparameter tuning with Optuna
Optimizes Temporal Fusion Transformer for F1 predictions
"""

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from implement_temporal_fusion import F1DataModule, F1PositionAccuracy
import json
from datetime import datetime

print("🔧 TFT Hyperparameter Tuning with Optuna")
print("=" * 60)

class OptunaTuning:
    def __init__(self):
        self.data_module = F1DataModule()
        self.data_module.prepare_data()
        self.data_module.setup()
        
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        
        # Suggest hyperparameters
        params = {
            # Architecture
            'hidden_size': trial.suggest_int('hidden_size', 16, 128, step=16),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 4),
            'attention_head_size': trial.suggest_int('attention_head_size', 1, 8),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
            'hidden_continuous_size': trial.suggest_int('hidden_continuous_size', 8, 64, step=8),
            
            # Learning
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.01, 1.0),
            
            # Loss configuration
            'quantiles': [0.1, 0.5, 0.9],  # Fixed for consistency
        }
        
        # Create model with suggested parameters
        model = TemporalFusionTransformer.from_dataset(
            self.data_module.training,
            hidden_size=params['hidden_size'],
            lstm_layers=params['lstm_layers'],
            attention_head_size=params['attention_head_size'],
            dropout=params['dropout'],
            hidden_continuous_size=params['hidden_continuous_size'],
            learning_rate=params['learning_rate'],
            loss=QuantileLoss(quantiles=params['quantiles']),
            logging_metrics=[F1PositionAccuracy()],
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        # Create trainer with pruning callback
        trainer = pl.Trainer(
            max_epochs=30,  # Shorter for tuning
            accelerator='auto',
            devices=1,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor='val_loss'),
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
            ],
            gradient_clip_val=params['gradient_clip_val'],
            enable_progress_bar=False,  # Disable for cleaner output
            logger=False,  # Disable for tuning
        )
        
        # Train model
        trainer.fit(model, datamodule=self.data_module)
        
        # Return validation loss for optimization
        val_loss = trainer.callback_metrics['val_loss'].item()
        
        # Also track position accuracy
        if 'val_position_accuracy' in trainer.callback_metrics:
            val_accuracy = trainer.callback_metrics['val_position_accuracy'].item()
            trial.set_user_attr('position_accuracy', val_accuracy)
        
        return val_loss
    
    def run_optimization(self, n_trials=50):
        """Run hyperparameter optimization"""
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(),
            study_name='tft_f1_optimization'
        )
        
        # Optimize
        print(f"🔄 Running {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials)
        
        # Print results
        print("\n" + "="*60)
        print("📊 Optimization Results")
        print("="*60)
        
        print(f"\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Val Loss): {trial.value:.4f}")
        if 'position_accuracy' in trial.user_attrs:
            print(f"  Position Accuracy: {trial.user_attrs['position_accuracy']:.2f}%")
        
        print(f"\nBest params:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        
        # Save results
        results = {
            'best_params': trial.params,
            'best_value': trial.value,
            'best_accuracy': trial.user_attrs.get('position_accuracy', None),
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat(),
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'accuracy': t.user_attrs.get('position_accuracy', None)
                }
                for t in study.trials
            ]
        }
        
        with open('tft_hyperparameter_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Saved results to tft_hyperparameter_results.json")
        
        # Create importance plot
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html('tft_param_importance.html')
            print(f"📊 Saved parameter importance plot to tft_param_importance.html")
        except:
            pass
        
        return study
    
    def train_best_model(self, best_params):
        """Train final model with best parameters"""
        print("\n🏆 Training final model with best parameters...")
        
        # Create model with best parameters
        model = TemporalFusionTransformer.from_dataset(
            self.data_module.training,
            hidden_size=best_params['hidden_size'],
            lstm_layers=best_params['lstm_layers'],
            attention_head_size=best_params['attention_head_size'],
            dropout=best_params['dropout'],
            hidden_continuous_size=best_params['hidden_continuous_size'],
            learning_rate=best_params['learning_rate'],
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            logging_metrics=[F1PositionAccuracy()],
            log_interval=10,
        )
        
        # Train with full epochs
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    filename='tft-best-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=1,
                    mode='min'
                )
            ],
            gradient_clip_val=best_params['gradient_clip_val'],
        )
        
        trainer.fit(model, datamodule=self.data_module)
        
        # Test
        test_results = trainer.test(model, datamodule=self.data_module)
        
        print(f"\n✅ Final test results: {test_results}")
        
        return model, trainer


if __name__ == "__main__":
    # Initialize tuner
    tuner = OptunaTuning()
    
    # Run optimization
    study = tuner.run_optimization(n_trials=50)
    
    # Train best model
    best_params = study.best_trial.params
    model, trainer = tuner.train_best_model(best_params)
    
    print("\n✅ Task TFT-005 complete!")
    print("🚀 Best model saved and ready for production integration")