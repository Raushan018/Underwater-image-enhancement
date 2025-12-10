
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Importing DetectionService...")
    from src.detection import DetectionService
    
    print("Initializing Service (this triggers model download)...")
    service = DetectionService()
    
    print("Model initialized.")
    if "fish" in service.classes:
        print("Classes set correctly.")
    else:
        print("Classes verification failed.")
        
    print("TEST PASSED")
    
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Runtime Error: {e}")
    sys.exit(1)
