from .env_wrapper import EnvWrapper
from .dm_env_wrapper import DMSuiteEnvWrapper


DMSUITE_ENVIRONMENTS = ['humanoid-run']


def create_env_wrapper(config):
    if config['env'] in DMSUITE_ENVIRONMENTS:
        return DMSuiteEnvWrapper(config['env'])
    return EnvWrapper(config['env'])
