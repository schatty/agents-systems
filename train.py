from models.td5.engine import ExperimentEngine


if __name__ == "__main__":
        CONFIG_PATH = 'experiments/td5/td5_bipedal.yml'
        engine = ExperimentEngine(CONFIG_PATH)
        engine.run()