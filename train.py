from models.td3.engine import ExperimentEngine


if __name__ == "__main__":
        CONFIG_PATH = 'experiments/td3/td3_bipedal.yml'
        engine = ExperimentEngine(CONFIG_PATH)
        engine.run()