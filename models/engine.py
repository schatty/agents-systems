from .ddpg.engine import Engine as DDPGEngine
from .d3pg.engine import Engine as D3PGEngine
from .d4pg.engine import Engine as D4PGEngine


def load_engine(config):
    print(f"Loading {config['model']} for {config['env']}.")
    if config["model"] == "ddpg":
        return DDPGEngine(config)
    if config["model"] == "d3pg":
        return D3PGEngine(config)
    if config["model"] == "d4pg":
        return D4PGEngine(config)
