from models.trainer import Trainer

config = {
    "agent": "A3C",
    "episodes": 100
}

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()