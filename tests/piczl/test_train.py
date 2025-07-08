import pytest
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from piczl.execute.train import train_new_models

def test_train_function():
    train_new_models()
