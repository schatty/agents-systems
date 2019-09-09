from .env_wrapper import EnvWrapper


class BipedalWalker(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config['env'])
        self.config = config

