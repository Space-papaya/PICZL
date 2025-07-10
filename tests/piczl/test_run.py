import os
import sys

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from piczl.execute.run import predict_redshifts

def test_run_function():
    result = predict_redshifts()
    print("predict_redshifts() returned:", result)
