#!/usr/bin/env python3
"""
Production Model Training Script
Handles missing dependencies gracefully
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models():
    """Train models with proper error handling"""
    
    # Check for required files
    required_files = [
        'load_real_f1_data.py',
        'load_smart_f1_data.py',
        'train_models_ensemble.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        sys.exit(1)
    
    try:
        # Try to import and run the ensemble training
        logger.info("Starting ensemble model training...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import the training module
        import train_models_ensemble
        
        # Run the training
        train_models_ensemble.main()
        
        logger.info("✅ Model training completed successfully")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Falling back to basic training...")
        
        # Try to run scripts/train_models_v2.py instead
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, 'scripts/train_models_v2.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✅ Fallback training completed")
            else:
                logger.error(f"Fallback training failed: {result.stderr}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Fallback training error: {e}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train_models()