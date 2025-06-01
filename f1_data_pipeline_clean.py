#!/usr/bin/env python3
"""
F1 Data Pipeline - CLEAN VERSION
Only ingests REAL F1 data, no mock/sample data
Uses FastF1 for official F1 timing data
"""

import fastf1
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RaceSession:
    """Clean race session data"""
    season: int
    round: int
    name: str
    date: str
    circuit: str
    country: str
    sprint_weekend: bool = False

@dataclass
class DriverData:
    """Clean driver data"""
    driver_ref: str  # 'max_verstappen'
    code: str        # 'VER'
    forename: str
    surname: str
    nationality: str
    dob: Optional[str] = None

@dataclass
class QualifyingResult:
    """Clean qualifying result"""
    race_id: int
    driver_id: int
    q1_time_ms: Optional[int]
    q2_time_ms: Optional[int]
    q3_time_ms: Optional[int]
    best_time_ms: int
    qualifying_position: int
    grid_position: int
    grid_penalty: int
    session_date: str

@dataclass
class RaceResult:
    """Clean race result"""
    race_id: int
    driver_id: int
    grid_position: int
    final_position: Optional[int]
    race_time_ms: Optional[int]
    fastest_lap_time_ms: Optional[int]
    points: int
    status: str
    laps_completed: int
    session_date: str


class F1DataPipelineClean:
    """Clean F1 data pipeline - REAL DATA ONLY"""
    
    def __init__(self, db_path: str = "f1_predictions_clean.db"):
        self.db_path = db_path
        
        # Enable FastF1 cache for performance
        cache_dir = Path("cache/fastf1")
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        # Official F1 2025 calendar
        self.f1_2025_calendar = [
            RaceSession(2025, 1, "Australian Grand Prix", "2025-03-16", "Albert Park", "Australia"),
            RaceSession(2025, 2, "Chinese Grand Prix", "2025-03-23", "Shanghai", "China"),
            RaceSession(2025, 3, "Japanese Grand Prix", "2025-04-13", "Suzuka", "Japan"),
            RaceSession(2025, 4, "Bahrain Grand Prix", "2025-04-20", "Bahrain", "Bahrain"),
            RaceSession(2025, 5, "Saudi Arabian Grand Prix", "2025-05-04", "Jeddah", "Saudi Arabia"),
            RaceSession(2025, 6, "Miami Grand Prix", "2025-05-11", "Miami", "USA"),
            RaceSession(2025, 7, "Emilia Romagna Grand Prix", "2025-05-18", "Imola", "Italy"),
            RaceSession(2025, 8, "Monaco Grand Prix", "2025-05-25", "Monaco", "Monaco"),
            RaceSession(2025, 9, "Spanish Grand Prix", "2025-06-01", "Barcelona", "Spain"),
            RaceSession(2025, 10, "Canadian Grand Prix", "2025-06-15", "Montreal", "Canada"),
            RaceSession(2025, 11, "Austrian Grand Prix", "2025-06-29", "Spielberg", "Austria"),
            RaceSession(2025, 12, "British Grand Prix", "2025-07-06", "Silverstone", "Great Britain"),
            RaceSession(2025, 13, "Hungarian Grand Prix", "2025-07-20", "Hungaroring", "Hungary"),
            RaceSession(2025, 14, "Belgian Grand Prix", "2025-07-27", "Spa-Francorchamps", "Belgium"),
            RaceSession(2025, 15, "Dutch Grand Prix", "2025-08-31", "Zandvoort", "Netherlands"),
            RaceSession(2025, 16, "Italian Grand Prix", "2025-09-07", "Monza", "Italy"),
            RaceSession(2025, 17, "Azerbaijan Grand Prix", "2025-09-21", "Baku", "Azerbaijan"),
            RaceSession(2025, 18, "Singapore Grand Prix", "2025-10-05", "Marina Bay", "Singapore"),
            RaceSession(2025, 19, "United States Grand Prix", "2025-10-19", "COTA", "USA"),
            RaceSession(2025, 20, "Mexican Grand Prix", "2025-10-26", "Mexico City", "Mexico"),
            RaceSession(2025, 21, "Brazilian Grand Prix", "2025-11-09", "Interlagos", "Brazil"),
            RaceSession(2025, 22, "Las Vegas Grand Prix", "2025-11-22", "Las Vegas", "USA"),
            RaceSession(2025, 23, "Qatar Grand Prix", "2025-11-30", "Lusail", "Qatar"),
            RaceSession(2025, 24, "Abu Dhabi Grand Prix", "2025-12-07", "Yas Marina", "UAE"),
        ]
    
    def setup_database(self):
        """Setup clean database schema"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Read and execute clean schema
            schema_path = Path(__file__).parent / "schema_production_clean.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                conn.executescript(schema_sql)
                logger.info("✅ Clean database schema created")
            else:
                logger.error(f"❌ Schema file not found: {schema_path}")
                
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
        finally:
            conn.close()
    
    def load_2025_calendar(self):
        """Load official 2025 F1 calendar"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for race in self.f1_2025_calendar:
                cursor.execute("""
                    INSERT OR REPLACE INTO races 
                    (id, season, round, name, date, circuit, country, sprint_weekend)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race.round,  # Use round as ID for 2025
                    race.season,
                    race.round,
                    race.name,
                    race.date,
                    race.circuit,
                    race.country,
                    race.sprint_weekend
                ))
            
            conn.commit()
            logger.info(f"✅ Loaded {len(self.f1_2025_calendar)} races for 2025 season")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Failed to load 2025 calendar: {e}")
            raise
        finally:
            conn.close()
    
    def load_real_drivers(self, season: int = 2024):
        """Load real F1 drivers from FastF1 data"""
        logger.info(f"📥 Loading real drivers for {season}...")
        
        try:
            # Get first race of the season to extract driver list
            session = fastf1.get_session(season, 1, 'R')
            session.load()
            
            drivers_data = []
            for _, driver in session.results.iterrows():
                driver_data = DriverData(
                    driver_ref=self._clean_driver_ref(driver['FullName']),
                    code=driver['Abbreviation'],
                    forename=driver['FirstName'] if 'FirstName' in driver else driver['FullName'].split()[0],
                    surname=driver['LastName'] if 'LastName' in driver else driver['FullName'].split()[-1],
                    nationality=driver.get('Country', 'Unknown')
                )
                drivers_data.append(driver_data)
            
            self._save_drivers(drivers_data)
            logger.info(f"✅ Loaded {len(drivers_data)} real drivers")
            
        except Exception as e:
            logger.error(f"❌ Failed to load real drivers: {e}")
            # Don't fallback to mock data - fail cleanly
            raise
    
    def load_real_qualifying_data(self, season: int, round: int) -> bool:
        """Load REAL qualifying data from FastF1"""
        logger.info(f"📥 Loading real qualifying data for {season} Round {round}...")
        
        try:
            # Get qualifying session
            session = fastf1.get_session(season, round, 'Q')
            session.load()
            
            if session.results.empty:
                logger.warning(f"⚠️ No qualifying data available for {season} Round {round}")
                return False
            
            qualifying_results = []
            for _, result in session.results.iterrows():
                # Convert times to milliseconds
                q1_ms = self._time_to_ms(result.get('Q1')) if pd.notna(result.get('Q1')) else None
                q2_ms = self._time_to_ms(result.get('Q2')) if pd.notna(result.get('Q2')) else None
                q3_ms = self._time_to_ms(result.get('Q3')) if pd.notna(result.get('Q3')) else None
                
                # Get best time
                best_time = min([t for t in [q1_ms, q2_ms, q3_ms] if t is not None])
                
                qualifying_result = QualifyingResult(
                    race_id=round,  # Assuming round = race_id for now
                    driver_id=self._get_driver_id(result['Abbreviation']),
                    q1_time_ms=q1_ms,
                    q2_time_ms=q2_ms,
                    q3_time_ms=q3_ms,
                    best_time_ms=best_time,
                    qualifying_position=result['Position'],
                    grid_position=result.get('GridPosition', result['Position']),
                    grid_penalty=0,  # Would need additional data for penalties
                    session_date=session.date.isoformat()
                )
                qualifying_results.append(qualifying_result)
            
            self._save_qualifying_results(qualifying_results)
            logger.info(f"✅ Loaded {len(qualifying_results)} real qualifying results")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load qualifying data: {e}")
            return False
    
    def load_real_race_results(self, season: int, round: int) -> bool:
        """Load REAL race results from FastF1"""
        logger.info(f"📥 Loading real race results for {season} Round {round}...")
        
        try:
            # Get race session
            session = fastf1.get_session(season, round, 'R')
            session.load()
            
            if session.results.empty:
                logger.warning(f"⚠️ No race results available for {season} Round {round}")
                return False
            
            race_results = []
            for _, result in session.results.iterrows():
                # Handle race time
                race_time_ms = self._time_to_ms(result.get('Time')) if pd.notna(result.get('Time')) else None
                fastest_lap_ms = self._time_to_ms(result.get('FastestLapTime')) if pd.notna(result.get('FastestLapTime')) else None
                
                # Determine status
                status = 'Finished' if pd.notna(result.get('Position')) else 'DNF'
                
                race_result = RaceResult(
                    race_id=round,
                    driver_id=self._get_driver_id(result['Abbreviation']),
                    grid_position=result.get('GridPosition', 20),
                    final_position=result.get('Position') if pd.notna(result.get('Position')) else None,
                    race_time_ms=race_time_ms,
                    fastest_lap_time_ms=fastest_lap_ms,
                    points=result.get('Points', 0),
                    status=status,
                    laps_completed=result.get('Laps', 0),
                    session_date=session.date.isoformat()
                )
                race_results.append(race_result)
            
            self._save_race_results(race_results)
            logger.info(f"✅ Loaded {len(race_results)} real race results")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load race results: {e}")
            return False
    
    def _clean_driver_ref(self, full_name: str) -> str:
        """Convert full name to driver reference"""
        return full_name.lower().replace(' ', '_').replace('-', '_')
    
    def _time_to_ms(self, time_obj) -> Optional[int]:
        """Convert FastF1 time object to milliseconds"""
        if time_obj is None or pd.isna(time_obj):
            return None
        
        try:
            # FastF1 times are pandas Timedelta objects
            return int(time_obj.total_seconds() * 1000)
        except:
            return None
    
    def _get_driver_id(self, code: str) -> int:
        """Get driver ID from code"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM drivers WHERE code = ?", (code,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def _save_drivers(self, drivers: List[DriverData]):
        """Save drivers to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for i, driver in enumerate(drivers, 1):
                cursor.execute("""
                    INSERT OR REPLACE INTO drivers 
                    (id, driver_ref, code, forename, surname, nationality)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (i, driver.driver_ref, driver.code, driver.forename, driver.surname, driver.nationality))
            
            conn.commit()
        finally:
            conn.close()
    
    def _save_qualifying_results(self, results: List[QualifyingResult]):
        """Save qualifying results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for result in results:
                if result.driver_id:  # Only save if we have a valid driver ID
                    cursor.execute("""
                        INSERT OR REPLACE INTO qualifying_results 
                        (race_id, driver_id, q1_time_ms, q2_time_ms, q3_time_ms, 
                         best_time_ms, qualifying_position, grid_position, grid_penalty, session_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.race_id, result.driver_id, result.q1_time_ms, result.q2_time_ms,
                        result.q3_time_ms, result.best_time_ms, result.qualifying_position,
                        result.grid_position, result.grid_penalty, result.session_date
                    ))
            
            conn.commit()
        finally:
            conn.close()
    
    def _save_race_results(self, results: List[RaceResult]):
        """Save race results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for result in results:
                if result.driver_id:  # Only save if we have a valid driver ID
                    cursor.execute("""
                        INSERT OR REPLACE INTO race_results 
                        (race_id, driver_id, grid_position, final_position, race_time_ms,
                         fastest_lap_time_ms, points, status, laps_completed, session_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.race_id, result.driver_id, result.grid_position, result.final_position,
                        result.race_time_ms, result.fastest_lap_time_ms, result.points,
                        result.status, result.laps_completed, result.session_date
                    ))
            
            conn.commit()
        finally:
            conn.close()


def main():
    """Load real F1 data pipeline"""
    logger.info("🚀 Starting F1 Clean Data Pipeline...")
    
    pipeline = F1DataPipelineClean()
    
    try:
        # 1. Setup clean database
        pipeline.setup_database()
        
        # 2. Load 2025 calendar
        pipeline.load_2025_calendar()
        
        # 3. Load real drivers (from 2024 data)
        pipeline.load_real_drivers(2024)
        
        # 4. Load some 2024 historical data for training
        logger.info("📥 Loading 2024 historical data for training...")
        for round_num in range(1, 6):  # First 5 races of 2024
            success_qual = pipeline.load_real_qualifying_data(2024, round_num)
            success_race = pipeline.load_real_race_results(2024, round_num)
            
            if success_qual and success_race:
                logger.info(f"✅ Loaded Round {round_num} data")
            else:
                logger.warning(f"⚠️ Partial data for Round {round_num}")
        
        logger.info("🎉 Clean data pipeline completed successfully!")
        logger.info("✅ No mock data was used - all data is real F1 data")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()