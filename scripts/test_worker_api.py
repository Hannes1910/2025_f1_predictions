#!/usr/bin/env python3
"""
Test the Cloudflare Worker API endpoints locally
"""

import urllib.request
import urllib.parse
import json
import time

# Worker is running on localhost:64362 (from previous output)
BASE_URL = "http://localhost:64362"

def test_api_endpoint(endpoint, method="GET", data=None, description=""):
    """Test an API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"🔗 Testing {method} {endpoint}")
        if description:
            print(f"   {description}")
        
        if method == "GET":
            req = urllib.request.Request(url, method="GET")
        else:
            req_data = json.dumps(data).encode() if data else None
            req = urllib.request.Request(url, data=req_data, method=method)
            req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                result = json.loads(response.read().decode())
                print(f"   ✅ Success: {response.status}")
                if isinstance(result, dict) and 'predictions' in result:
                    print(f"   📊 Found {len(result['predictions'])} predictions")
                elif isinstance(result, list):
                    print(f"   📊 Found {len(result)} items")
                return True
            else:
                print(f"   ❌ Error: {response.status}")
                return False
                
    except urllib.error.HTTPError as e:
        print(f"   ❌ HTTP Error: {e.code}")
        if e.code == 404:
            print("   💡 Endpoint not found - may not be implemented yet")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_worker_endpoints():
    """Test all worker API endpoints"""
    print("🧪 Testing Cloudflare Worker API Endpoints\n")
    
    endpoints = [
        ("/", "GET", None, "Root endpoint - should show API info"),
        ("/api/health", "GET", None, "Health check endpoint"),
        ("/api/races", "GET", None, "Get all races"),
        ("/api/drivers", "GET", None, "Get all drivers"), 
        ("/api/predictions", "GET", None, "Get all predictions"),
        ("/api/predictions/latest", "GET", None, "Get latest predictions"),
    ]
    
    passed = 0
    total = len(endpoints)
    
    for endpoint, method, data, desc in endpoints:
        if test_api_endpoint(endpoint, method, data, desc):
            passed += 1
        print()
        time.sleep(0.5)  # Small delay between requests
    
    print("=" * 50)
    print(f"📊 API Test Results: {passed}/{total} endpoints working")
    
    if passed >= total // 2:
        print("✅ Worker API is responding!")
        print("\n💡 Next steps:")
        print("1. 🔄 Add sample data to D1 database")
        print("2. 🚀 Deploy to production: wrangler deploy")
        print("3. 🌐 Test frontend integration")
    else:
        print("⚠️  Some endpoints may not be implemented yet")
    
    return passed >= total // 2

def main():
    print("🔗 Cloudflare Worker API Test\n")
    print("📡 Testing local worker on http://localhost:64362")
    print("💡 Make sure wrangler dev is running in another terminal\n")
    
    try:
        # Quick connectivity test
        test_api_endpoint("/", "GET", None, "Testing basic connectivity")
        print()
        
        # Run full endpoint tests
        return test_worker_endpoints()
        
    except Exception as e:
        print(f"❌ Failed to connect to worker: {e}")
        print("💡 Make sure to run: cd packages/worker && npx wrangler dev --local")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)