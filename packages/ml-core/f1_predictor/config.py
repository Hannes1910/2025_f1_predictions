from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

class ModelType(Enum):
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"

@dataclass
class PredictorConfig:
    """Configuration for F1 predictor"""
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    features: List[str] = None
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 5
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Feature flags
    use_weather: bool = True
    use_team_performance: bool = True
    use_sector_times: bool = True
    use_historical_performance: bool = True
    
    # API keys
    weather_api_key: Optional[str] = None
    
    # Cache settings
    cache_dir: str = "f1_cache"
    
    def __post_init__(self):
        if self.features is None:
            self.features = self._get_default_features()
    
    def _get_default_features(self) -> List[str]:
        """Get default feature set based on configuration"""
        features = ["QualifyingTime (s)"]
        
        if self.use_sector_times:
            features.extend(["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"])
        
        if self.use_weather:
            features.extend(["RainProbability", "Temperature"])
        
        if self.use_team_performance:
            features.extend(["TeamPerformanceScore", "SeasonPoints"])
        
        if self.use_historical_performance:
            features.extend(["AvgPosition", "RecentForm"])
        
        return features

@dataclass
class RaceData:
    """Data for a single race"""
    race_id: int
    race_name: str
    circuit: str
    date: str
    qualifying_data: Dict[str, float]
    weather_data: Optional[Dict[str, float]] = None
    team_points: Optional[Dict[str, int]] = None

@dataclass
class Prediction:
    """Single driver prediction"""
    driver_code: str
    driver_name: str
    predicted_position: int
    predicted_time: float
    confidence: float
    
@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mae: float
    rmse: float
    accuracy: float
    feature_importance: Dict[str, float]