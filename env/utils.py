from .env_wrapper import EnvWrapper


def create_env_wrapper(config):
    return EnvWrapper(config['env'])
