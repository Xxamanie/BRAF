#!/usr/bin/env python3
"""
Debug NEXUS7 Integration
"""

import sys
import traceback

try:
    print("Importing nexus7_integration module...")
    import research.nexus7_integration as ni
    print(f"Module imported successfully")
    print(f"Module attributes: {dir(ni)}")
    
    print("\nTrying to access NEXUS7Integration class...")
    if hasattr(ni, 'NEXUS7Integration'):
        print("NEXUS7Integration class found!")
        cls = getattr(ni, 'NEXUS7Integration')
        print(f"Class: {cls}")
        
        print("Creating instance...")
        instance = cls()
        print(f"Instance created: {instance}")
    else:
        print("NEXUS7Integration class NOT found!")
        print("Available attributes:")
        for attr in dir(ni):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()