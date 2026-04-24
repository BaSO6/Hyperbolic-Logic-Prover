import sys
import os
import time

# ---------------------------------------------------------
# [CRITICAL FIX] Automatically add project root to Python search path
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)  # src/system2
src_dir = os.path.dirname(current_dir)           # src
project_root = os.path.dirname(src_dir)          # Project Root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports can be performed normally
try:
    from src.system2.lean_interaction import LeanEnv
except ImportError as e:
    print(f"❌ Python Path Error: {e}")
    print(f"    Debug Info: Project Root detected at: {project_root}")
    sys.exit(1)

# ---------------------------------------------------------
# Test Logic
# ---------------------------------------------------------
def test_minimal_env():
    print(f"📂 Project Root: {project_root}")
    print("🚀 Initializing LeanEnv (Minimal Test)...")
    
    try:
        # 1. Startup
        env = LeanEnv(project_root, verbose=True)
        
        # 2. Attempt import (if you created Search.lean, you can also try "import Search" here)
        print("\n[Step 1] Sending Import...")
        # Note: If you use the v6.0 wrapper I provided, the import here will be very fast
        res = env.run_command("import Mathlib.Tactic", timeout=60)
        print(f"    Response: {res}")
        
        # 3. Open namespaces (prevents "failed to synthesize" errors)
        print("\n[Step 2] Opening Namespaces...")
        env.run_command("open Nat Real Rat BigOperators")
        
        # 4. Test basic theorem (if there are issues, it will report "failed to synthesize" here)
        print("\n[Step 3] Proving 1 + 1 = 2...")
        res = env.run_command("example : 1 + 1 = 2 := by rfl", timeout=10)
        print(f"    Response: {res}")
        
        if "messages" not in res and "env" in res:
            print("\n🎉🎉🎉 SUCCESS: Environment is perfectly healthy! 🎉🎉🎉")
        elif "messages" in res:
            # Check for error messages
            errors = [m for m in res['messages'] if m['severity'] == 'error']
            if not errors:
                print("\n🎉🎉🎉 SUCCESS: Environment is healthy (Warnings ignored) 🎉🎉🎉")
            else:
                print("\n💀 FAILED: Lean reported errors.")
                
        env.close()
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_minimal_env()