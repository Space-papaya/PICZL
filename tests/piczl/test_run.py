import os
import sys

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add src to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

# Verify path added
print("sys.path[0]:", sys.path[0])  # should be path/to/project/src

# Try import
try:
    from piczl.execute.run import predict_redshifts
except ImportError as e:
    print("Import failed:", e)
    raise

def test_run_function():
    result = predict_redshifts()
    print("predict_redshifts() returned:", result)
