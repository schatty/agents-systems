from .ddpg.engine import Engine as DDPGEngine
from .d3pg.engine import Engine as D3PGEngine
from .d4pg.engine import Engine as D4PGEngine
from .td3.engine import Engine as TD3Engine
from .td3_q3.engine import Engine as TD3Q3Engine
from .td3_q4.engine import Engine as TD3Q4Engine


def load_engine(config):
    print(f"Loading {config['model']} for {config['env']}.")
    if config["model"] == "ddpg":
        return DDPGEngine(config)
    if config["model"] == "d3pg":
        return D3PGEngine(config)
    if config["model"] == "d4pg":
        return D4PGEngine(config)
    if config["model"] == "td3":
        return TD3Engine(config)
    if config["model"] == "td3_q3":
        return TD3Q3Engine(config)
    if config["model"] == "td3_q4":
        return TD3Q4Engine(config)
