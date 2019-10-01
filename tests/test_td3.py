import unittest
from models.td3.engine import ExperimentEngine


class TestsPendulumD3PG(unittest.TestCase):

    def test_d3pg_train(self):
        # CONFIG_PATH = 'experiments/td3/td3_bipedal.yml'
        CONFIG_PATH = 'experiments/td3/td3_learn_to_move.yml'
        engine = ExperimentEngine(CONFIG_PATH)
        engine.run()