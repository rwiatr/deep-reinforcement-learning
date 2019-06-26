import unityagents

from drl.env import Env


class SingleBrainEnv(Env):

    def __init__(self, env_path, train_mode=True):
        self.env = unityagents.UnityEnvironment(file_name=env_path)
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_dim = self.brain.vector_action_space_size
        self.state_dim = len(env_info.vector_observations[0])
        self.train_mode = train_mode

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        return env_info.vector_observations, env_info.rewards, env_info.local_done, None

    def get_num_agents(self):
        return self.num_agents

    def close(self):
        self.env.close()

    def get_action_dim(self):
        return self.action_dim

    def get_state_dim(self):
        return self.state_dim
