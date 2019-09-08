import unittest
from models.d3pg.engine import ExperimentEngine


class TestsPendulumD3PG(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'experiments/ddpg/ddpg_pendulum.yml'
        engine = ExperimentEngine(CONFIG_PATH)
        engine.run()