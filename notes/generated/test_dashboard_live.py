"""
Live test of dashboard statistics endpoint.

Run this with the backend server running to test the actual endpoint.
"""

import requests
import json

def test_dashboard_statistics():
    """Test the dashboard statistics endpoint."""
    url = "http://localhost:8000/api/v1/analysis/statistics"
    
    print(f"Testing endpoint: {url}")
    print("-" * 60)
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS - Response received")
            print("\nResponse structure:")
            print(json.dumps(data, indent=2))
            
            if data.get("success"):
                stats = data.get("statistics", {})
                print("\n" + "=" * 60)
                print("DASHBOARD STATISTICS SUMMARY")
                print("=" * 60)
                print(f"Total Analyses: {stats.get('total_analyses', 0)}")
                print(f"  - Gait Analyses: {stats.get('total_gait_analyses', 0)}")
                print(f"  - GAVD Datasets: {stats.get('total_gavd_datasets', 0)}")
                print(f"\nGAVD Datasets:")
                print(f"  - Completed: {stats.get('gavd_completed', 0)}")
                print(f"  - Processing: {stats.get('gavd_processing', 0)}")
                print(f"  - Total Sequences: {stats.get('total_gavd_sequences', 0)}")
                print(f"  - Total Frames: {stats.get('total_gavd_frames', 0)}")
                print(f"\nGait Analysis:")
                print(f"  - Normal Patterns: {stats.get('normal_patterns', 0)}")
                print(f"  - Abnormal Patterns: {stats.get('abnormal_patterns', 0)}")
                print(f"  - Avg Confidence: {stats.get('avg_confidence', 0):.1f}%")
                
                recent = stats.get('recent_analyses', [])
                print(f"\nRecent Analyses: {len(recent)} items")
                for i, item in enumerate(recent[:5], 1):
                    item_type = item.get('type', 'unknown')
                    if item_type == 'gavd_dataset':
                        print(f"  {i}. [GAVD] {item.get('filename', 'Unknown')} - {item.get('status', 'unknown')}")
                    else:
                        print(f"  {i}. [GAIT] Analysis {item.get('analysis_id', 'unknown')[:8]}... - {item.get('status', 'unknown')}")
                
                print("\n✅ All data loaded successfully!")
                return True
            else:
                print("\n❌ Response indicates failure")
                return False
        else:
            print(f"\n❌ ERROR - Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR - Could not connect to server")
        print("Make sure the backend server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"\n❌ ERROR - {str(e)}")
        return False


if __name__ == "__main__":
    success = test_dashboard_statistics()
    exit(0 if success else 1)
