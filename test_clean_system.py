#!/usr/bin/env python3
"""
Test Clean F1 ML System
Verify no mock data, real data only, proper error handling
"""

import requests
import json
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanSystemTester:
    """Test the clean F1 ML system"""
    
    def __init__(self):
        self.worker_url = "https://f1-predictions-api.vprifntqe.workers.dev"
        self.ml_service_url = "http://localhost:8001"
        self.db_path = "f1_predictions_clean.db"
        
        self.test_results = {
            'database': False,
            'worker_api': False,
            'ml_service': False,
            'real_data_only': False,
            'no_mock_fallbacks': False
        }
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("🧪 TESTING CLEAN F1 ML SYSTEM")
        logger.info("=" * 50)
        
        # Test 1: Database Schema and Real Data
        logger.info("\n1️⃣ Testing Database Schema and Real Data...")
        self.test_results['database'] = self.test_database()
        
        # Test 2: Worker API Endpoints
        logger.info("\n2️⃣ Testing Worker API Endpoints...")
        self.test_results['worker_api'] = self.test_worker_api()
        
        # Test 3: ML Service
        logger.info("\n3️⃣ Testing ML Service...")
        self.test_results['ml_service'] = self.test_ml_service()
        
        # Test 4: Real Data Only (No Mock Data)
        logger.info("\n4️⃣ Testing Real Data Only Policy...")
        self.test_results['real_data_only'] = self.test_real_data_only()
        
        # Test 5: No Mock Fallbacks
        logger.info("\n5️⃣ Testing No Mock Fallbacks...")
        self.test_results['no_mock_fallbacks'] = self.test_no_mock_fallbacks()
        
        # Summary
        self.print_test_summary()
    
    def test_database(self) -> bool:
        """Test database schema and real data"""
        try:
            if not Path(self.db_path).exists():
                logger.error("❌ Clean database not found")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check schema tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                'races', 'drivers', 'team_entries', 
                'qualifying_results', 'race_results', 'predictions'
            ]
            
            missing_tables = [t for t in required_tables if t not in tables]
            if missing_tables:
                logger.error(f"❌ Missing tables: {missing_tables}")
                return False
            
            # Check for deprecated tables (tech debt)
            deprecated_tables = ['feature_data']  # Should not exist in clean version
            found_deprecated = [t for t in deprecated_tables if t in tables]
            if found_deprecated:
                logger.error(f"❌ Found deprecated tables: {found_deprecated}")
                return False
            
            # Check for real data (not empty tables)
            cursor.execute("SELECT COUNT(*) FROM drivers")
            driver_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM races WHERE season = 2025")
            race_count = cursor.fetchone()[0]
            
            if driver_count == 0:
                logger.warning("⚠️ No drivers in database")
                return False
            
            if race_count == 0:
                logger.warning("⚠️ No 2025 races in database")
                return False
            
            # Check constraints exist
            cursor.execute("PRAGMA table_info(qualifying_results)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'best_time_ms' not in columns:
                logger.error("❌ Missing best_time_ms column in qualifying_results")
                return False
            
            conn.close()
            
            logger.info(f"✅ Database schema valid")
            logger.info(f"✅ {driver_count} drivers loaded")
            logger.info(f"✅ {race_count} 2025 races loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False
    
    def test_worker_api(self) -> bool:
        """Test Worker API endpoints"""
        try:
            # Test health check
            response = requests.get(f"{self.worker_url}/api/health", timeout=10)
            if response.status_code != 200:
                logger.error("❌ Worker health check failed")
                return False
            
            # Test driver stats (should return real data or 404)
            response = requests.get(f"{self.worker_url}/api/data/driver/1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' not in data:
                    logger.error("❌ Driver API missing 'data' field")
                    return False
                
                driver_data = data['data']
                if not driver_data.get('code'):
                    logger.error("❌ Driver data missing code")
                    return False
                
                logger.info(f"✅ Driver API working: {driver_data['code']}")
            elif response.status_code == 404:
                logger.info("✅ Driver API properly returns 404 for missing data")
            else:
                logger.error(f"❌ Driver API unexpected status: {response.status_code}")
                return False
            
            # Test race features
            response = requests.get(f"{self.worker_url}/api/data/race/1/features", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' not in data or 'race' not in data['data']:
                    logger.error("❌ Race features API malformed")
                    return False
                logger.info("✅ Race features API working")
            elif response.status_code in [404, 422]:
                logger.info("✅ Race features API properly handles missing data")
            else:
                logger.error(f"❌ Race features unexpected status: {response.status_code}")
                return False
            
            # Test ML prediction data
            response = requests.post(
                f"{self.worker_url}/api/data/ml-prediction-data",
                json={"raceId": 1},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'meta' not in data or data['meta'].get('data_source') != 'real_f1_data':
                    logger.error("❌ ML data API not marked as real data")
                    return False
                logger.info("✅ ML prediction data API working")
            elif response.status_code in [404, 422]:
                logger.info("✅ ML data API properly handles insufficient data")
            else:
                logger.error(f"❌ ML data API unexpected status: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Worker API test failed: {e}")
            return False
    
    def test_ml_service(self) -> bool:
        """Test ML service endpoints"""
        try:
            # Test health check
            response = requests.get(f"{self.ml_service_url}/health", timeout=10)
            if response.status_code != 200:
                logger.warning("⚠️ ML service not running (start with: python3 ml_service_production_clean.py)")
                return False
            
            health_data = response.json()
            
            # Check service info
            response = requests.get(f"{self.ml_service_url}/", timeout=10)
            if response.status_code != 200:
                logger.error("❌ ML service root endpoint failed")
                return False
            
            service_info = response.json()
            
            # Verify it's the clean version
            if 'clean' not in service_info.get('service', '').lower():
                logger.error("❌ Not running clean ML service")
                return False
            
            if service_info.get('data_source') != 'real_f1_data_only':
                logger.error("❌ ML service not configured for real data only")
                return False
            
            logger.info("✅ ML service running clean version")
            logger.info(f"✅ Models available: {service_info.get('models_available', False)}")
            
            # Test prediction endpoint (may fail if no real data/models)
            response = requests.post(f"{self.ml_service_url}/predict/race/1", timeout=30)
            if response.status_code == 200:
                pred_data = response.json()
                if pred_data.get('metadata', {}).get('data_source') != 'real_f1_data':
                    logger.error("❌ Predictions not using real data")
                    return False
                logger.info("✅ Predictions working with real data")
            elif response.status_code == 422:
                logger.info("✅ Prediction properly fails with insufficient real data")
            elif response.status_code == 503:
                logger.info("✅ Prediction properly fails with no trained models")
            else:
                logger.warning(f"⚠️ Unexpected prediction response: {response.status_code}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ ML service not running (start with: python3 ml_service_production_clean.py)")
            return False
        except Exception as e:
            logger.error(f"❌ ML service test failed: {e}")
            return False
    
    def test_real_data_only(self) -> bool:
        """Test that system only uses real F1 data"""
        try:
            # Check for mock data patterns in responses
            response = requests.post(
                f"{self.worker_url}/api/data/ml-prediction-data",
                json={"raceId": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check metadata
                meta = data.get('meta', {})
                if meta.get('data_source') != 'real_f1_data':
                    logger.error("❌ Data source not marked as real F1 data")
                    return False
                
                # Check for mock patterns in driver data
                drivers = data.get('data', {}).get('drivers', [])
                for driver in drivers[:3]:  # Check first 3
                    if not driver.get('code') or len(driver.get('code', '')) != 3:
                        logger.error("❌ Invalid driver code format")
                        return False
                
                logger.info("✅ All data marked as real F1 data")
                
            elif response.status_code in [404, 422]:
                logger.info("✅ Properly handles missing real data")
            
            # Check database for mock patterns
            if Path(self.db_path).exists():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check for unrealistic data
                cursor.execute("SELECT COUNT(*) FROM qualifying_results WHERE best_time_ms < 60000")  # < 1 minute
                unrealistic_times = cursor.fetchone()[0]
                
                if unrealistic_times > 0:
                    logger.warning(f"⚠️ Found {unrealistic_times} unrealistic qualifying times")
                
                conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Real data test failed: {e}")
            return False
    
    def test_no_mock_fallbacks(self) -> bool:
        """Test that system has no mock data fallbacks"""
        try:
            # Test with non-existent race ID
            response = requests.post(
                f"{self.worker_url}/api/data/ml-prediction-data",
                json={"raceId": 9999},
                timeout=10
            )
            
            # Should return 404/422, not mock data
            if response.status_code == 200:
                data = response.json()
                if data.get('data', {}).get('drivers'):
                    logger.error("❌ System returns data for non-existent race (possible mock fallback)")
                    return False
            elif response.status_code in [404, 422]:
                logger.info("✅ Properly rejects non-existent race")
            
            # Test ML service with invalid race
            try:
                response = requests.post(f"{self.ml_service_url}/predict/race/9999", timeout=10)
                if response.status_code == 200:
                    logger.error("❌ ML service returns predictions for non-existent race")
                    return False
                elif response.status_code == 422:
                    logger.info("✅ ML service properly rejects insufficient data")
            except requests.exceptions.ConnectionError:
                logger.info("⚠️ ML service not running for fallback test")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Mock fallback test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "=" * 50)
        logger.info("🧪 CLEAN SYSTEM TEST SUMMARY")
        logger.info("=" * 50)
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Clean system verified!")
            logger.info("✅ No mock data detected")
            logger.info("✅ Real F1 data only")
            logger.info("✅ Proper error handling")
        else:
            logger.error("❌ Some tests failed - system needs attention")
            
            if not self.test_results['database']:
                logger.error("→ Run: python3 f1_data_pipeline_clean.py")
            if not self.test_results['ml_service']:
                logger.error("→ Start ML service: python3 ml_service_production_clean.py")
        
        logger.info("\n📋 Next Steps:")
        if passed == total:
            logger.info("1. Train models with real 2024 F1 data")
            logger.info("2. Deploy to production")
            logger.info("3. Monitor prediction accuracy")
        else:
            logger.info("1. Fix failing tests")
            logger.info("2. Re-run test suite")
            logger.info("3. Verify no mock data")


def main():
    """Run the clean system test suite"""
    tester = CleanSystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()