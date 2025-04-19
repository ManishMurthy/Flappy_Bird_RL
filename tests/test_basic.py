import sys
import os

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

def test_import_environment():
    try:
        from src.flappy_env import FlappyBirdEnv
        assert True
    except ImportError:
        assert False, "Failed to import FlappyBirdEnv"

def test_import_agent():
    try:
        from src.dqn_agent import DQNAgent
        assert True
    except ImportError:
        assert False, "Failed to import DQNAgent"