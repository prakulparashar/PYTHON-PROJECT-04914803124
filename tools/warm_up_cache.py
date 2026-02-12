import os
from src.audit_engine import get_city_network, get_pois


districts = [
    "Central Delhi", "Central North Delhi", "East Delhi", "New Delhi", 
    "North Delhi", "North East Delhi", "North West Delhi", "Old Delhi", 
    "Outer North Delhi", "South Delhi", "South East Delhi", "South West Delhi", 
    "West Delhi"
]


amenities = ["hospital", "school", "supermarket", "pharmacy"]

def warm_up():
    print("Starting Batch Cache Audit for Delhi...")
    
    for district in districts:
        full_place_name = f"{district}, Delhi, India"
        print(f"\n--- Auditing: {district} ---")
        
        try:
            
            print(f"📦 Fetching Road Network...")
            get_city_network(full_place_name)
            
            for amenity in amenities:
                print(f"📍 Fetching {amenity} POIs...")
                get_pois(full_place_name, amenity)
                
        except Exception as e:
            print(f"⚠️ Failed to cache {district}: {e}")

    print("\nAll locations cached")

if __name__ == "__main__":
    warm_up()