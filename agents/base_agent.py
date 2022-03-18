from utils.replaybuffer import ReplayBuffer

# Agent interface
# Takes an environment (just so we can get some details from the environment like the number of observables and actions)
class BaseAgent(object):
    def __init__(self, config, network_manager):


        # Env config
        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        self.use_replay = config.use_replay
        if self.use_replay:
            self.replay_buffer = ReplayBuffer(config.buffer_size, config.random_seed)
        else:
            self.replay_buffer = None
        self.batch_size = config.batch_size
        self.warmup_steps = config.warmup_steps
        self.gamma = config.gamma

        # to log useful stuff within agent
        self.write_log = config.write_log
        self.write_plot = config.write_plot

        self.network_manager = network_manager
        self.config = config

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):

        if self.use_replay and self.replay_buffer.get_size() < self.warmup_steps:
            # Currently not using warmup steps
            raise NotImplementedError
        else:
            action = self.network_manager.take_action(state, is_train, is_start)
        return action

    def get_value(self, s, a):
        raise NotImplementedError
    
    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        if not is_truncated:

            # if using replay buffer
            if self.use_replay:
                if not is_terminal:
                    self.replay_buffer.add(state, action, reward, next_state, self.gamma)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, 0.0)
                self.learn()

            # if not using replay buffer
            else:
                if not is_terminal:
                    self.learn([state], [action], [reward], [next_state], [self.gamma])
                else:
                    self.learn([state], [action], [reward], [next_state], [0.0])

    def learn(self, state=None, action=None, reward=None, next_state=None, gamma=None):

        # if using replay, overwrite with batches
        if self.use_replay:
            if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
                state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)

                self.network_manager.update_network(state, action, next_state, reward, gamma)
        else:
            assert state is not None
            self.network_manager.update_network(state, action, next_state, reward, gamma)


    # Resets the agent between episodes. Should primarily be used to clear traces or other temporally linked parameters
    def reset(self):
        self.network_manager.reset()
