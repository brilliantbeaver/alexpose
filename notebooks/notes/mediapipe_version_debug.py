# MediaPipe Version Debugging Script
# This will help identify the exact issue with your MediaPipe installation

debug_cell = '''
# STEP 1: Check what MediaPipe you actually have
import sys
print("🔍 MEDIAPIPE DEBUGGING")
print("=" * 50)

try:
    import mediapipe as mp
    print(f"✅ MediaPipe imported successfully")
    print(f"📦 MediaPipe version: {mp.__version__}")
    print(f"📍 MediaPipe location: {mp.__file__}")
    
    # Check what attributes MediaPipe actually has
    print(f"\\n🔧 Available MediaPipe attributes:")
    mp_attrs = [attr for attr in dir(mp) if not attr.startswith('_')]
    for i, attr in enumerate(mp_attrs):
        print(f"  {i+1:2d}. {attr}")
    
    # Specifically check for 'solutions'
    if hasattr(mp, 'solutions'):
        print(f"\\n✅ mp.solutions EXISTS")
        solutions_attrs = [attr for attr in dir(mp.solutions) if not attr.startswith('_')]
        print(f"🔧 Available solutions:")
        for attr in solutions_attrs:
            print(f"  - {attr}")
    else:
        print(f"\\n❌ mp.solutions DOES NOT EXIST")
        print(f"🚨 This is the problem! You have a different MediaPipe version/package")
    
    # Check for alternative imports
    print(f"\\n🔍 Checking alternative MediaPipe imports...")
    
    # Try different import patterns
    alternatives = [
        "mediapipe.python.solutions",
        "mediapipe.framework.formats",
        "mediapipe.tasks.python.vision",
    ]
    
    for alt in alternatives:
        try:
            exec(f"import {alt}")
            print(f"✅ {alt} - Available")
        except ImportError as e:
            print(f"❌ {alt} - Not available: {e}")

except ImportError as e:
    print(f"❌ MediaPipe import failed: {e}")

print("\\n" + "=" * 50)
'''

print("COPY THIS INTO A JUPYTER CELL TO DEBUG:")
print("=" * 50)
print(debug_cell)