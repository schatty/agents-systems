class UnityEnvWrapper:
    def __init__(self, unity_env):
        self.env = unity_env
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.train_mode = True
        self.info = None

    def reset(self):
        """Return state."""
        self.info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.info.vector_observations[0]

    def step(self, action):
        """Return (next_state, reward, done) tuple. """
        self.info = self.env.step(action)[self.brain_name]
        next_state = self.info.vector_observations[0]
        reward = self.info.rewards[0]
        done = self.info.local_done[0]
        return next_state, reward, done

    def eval_mode(self):
        self.train_mode = False

    def train_mode(self):
        self.train_mode = True

    def close(self):
        self.env.close()

    @property
    def action_dim(self):
        return self.brain.vector_action_space_size

    @property
    def state_dim(self):
        if self.info is None:
            print("Reset environment to get access to the state size.")
            return
        return len(self.info.vector_observations[0])

    def render(self):
        return self.info.visual_observations[0]
