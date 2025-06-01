# Experimental Prediction Scripts Archive

This folder contains experimental prediction scripts that were used to test different approaches during development. These are kept for reference but are NOT used in production.

## Production Models
The production prediction system uses:
- `train_models_v2.py` - Main production model training script
- `ml_package/` - Production ML package with advanced features

## Experimental Scripts

### prediction1.py
- **Purpose**: Initial simple model using only qualifying times
- **Features**: Basic linear regression with qualifying position
- **Key Learning**: Too simplistic, needed more features

### prediction2.py
- **Purpose**: Added team performance factors
- **Features**: Qualifying times + constructor standings
- **Key Learning**: Team performance is important but not sufficient

### prediction2_nochange.py
- **Purpose**: Variant testing static predictions
- **Key Learning**: Baseline comparison

### prediction2_olddrivers.py
- **Purpose**: Testing with historical driver data
- **Key Learning**: Driver history patterns

### prediction3.py
- **Purpose**: Introduced weather data integration
- **Features**: Added OpenWeatherMap API for weather conditions
- **Key Learning**: Weather significantly impacts predictions

### prediction4.py
- **Purpose**: Added sector time analysis
- **Features**: S1, S2, S3 sector performance from FastF1
- **Key Learning**: Sector times provide granular performance insights

### prediction5.py
- **Purpose**: Introduced driver consistency metrics
- **Features**: Position variance, historical performance
- **Key Learning**: Consistency is a strong predictor

### prediction6.py
- **Purpose**: Added wet weather performance factors
- **Features**: Driver-specific wet performance ratings
- **Key Learning**: Wet conditions require specialized modeling

### prediction7.py
- **Purpose**: Circuit-specific features
- **Features**: Track characteristics, historical circuit performance
- **Key Learning**: Track-specific patterns matter

### prediction8.py
- **Purpose**: Monaco GP specialized model
- **Features**: Clean air race pace, position change patterns
- **Key Learning**: Some tracks need specialized approaches

## Key Insights from Experiments

1. **Feature Importance** (from most to least important):
   - Qualifying performance (35%)
   - Team/car performance (25%)
   - Weather conditions (15%)
   - Driver consistency (12%)
   - Circuit history (8%)
   - Other factors (5%)

2. **Model Evolution**:
   - Started with Linear Regression
   - Moved to Random Forest
   - Settled on Gradient Boosting for production

3. **Data Sources**:
   - FastF1 API proved most reliable
   - Weather data essential for accuracy
   - Real-time telemetry would be next improvement

These experiments led to the current production model in `train_models_v2.py` which combines the best features and learnings from all these approaches.