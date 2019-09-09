from models.d3pg.engine import ExperimentEngine


if __name__ == "__main__":
        CONFIG_PATH = 'experiments/ddpg/ddpg_learn_to_move.yml'
        engine = ExperimentEngine(CONFIG_PATH)
        engine.run()