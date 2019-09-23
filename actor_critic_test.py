from models.trainer import Trainer
from utils.misc import read_config


if __name__ == "__main__":
    config = read_config("experiments/actor_critic/lunar_lander.yml")
    trainer = Trainer(config)
    trainer.train()