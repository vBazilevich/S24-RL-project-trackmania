import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server

from tmrl.training_offline import TrainingOffline

import numpy as np
import os


epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"

memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.
env_cls = cfg_obj.ENV_CLS
device_worker = 'cuda' if cfg.CUDA_INFERENCE else 'cpu'


# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT
img_grayscale = cfg.GRAYSCALE
imgs_buf_len = cfg.IMG_HIST_LEN

act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# =====================================================================
# CUSTOM MODEL
# =====================================================================
# The following constants are from the Spinup implementation of SAC
# that we simply copy/paste and adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20

from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        do = nn.Dropout1d(0.15) if j < len(sizes) - 2 else nn.Identity()
        layers += [nn.Linear(sizes[j], sizes[j + 1]), do, act()]
    return nn.Sequential(*layers)

# Let us now define a module that will be the main building block of both our actor and critic:
class RNNNet(nn.Module):
    def __init__(self, q_net):
        """
        Args:
            q_net (bool): indicates whether this neural net is a critic network
        """
        super().__init__()
        self.q_net = q_net

        self.rnn = nn.LSTM(19, 64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)

        # The MLP input will be formed of:
        # - the flattened RNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 10 if self.q_net else 7
        self.mlp_input_features = 2 * self.rnn.num_layers * self.rnn.hidden_size + float_features

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later in the tutorial;
        # thus, the output dimensionality is  1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        In Pytorch, the forward function is where our neural network computes its output from its input.

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, lidar, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, lidar, act1, act2 = x

        lidar = lidar.reshape(-1, 4, 19)
        self.rnn.flatten_parameters()
        _, (x, _) = self.rnn(lidar)

        x = x.reshape(-1, 2 * self.rnn.num_layers * self.rnn.hidden_size)


        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class MyActorModule(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """
    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0]  # dimensionality of actions
        act_limit = action_space.high[0]  # maximum amplitude of actions
        self.net = RNNNet(q_net=False)
        # The policy output layer, which samples actions stochastically in a gaussian, with means...:
        self.mu_layer = nn.Linear(256, dim_act)
        # ... and log standard deviations:
        self.log_std_layer = nn.Linear(256, dim_act)
        # We will squash this within the action space thanks to a tanh final activation:
        self.act_limit = act_limit

    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the output action of our policy from the input observation.

        The whole point of deep RL is to train our policy network (actor) such that it outputs relevant actions.
        Training per-se will also rely on a critic network, but this is not part of the trained policy.
        Thus, our ActorModule will only implement the actor.

        Args:
            obs: the observation from the Gymnasium environment (when using TorchActorModule this is a torch.Tensor)
            test (bool): this is True for test episodes (deployment) and False for training episodes;
                in SAC, this enables us to sample randomly during training and deterministically at test-time.
            compute_logprob (bool): SAC will set this to True to retrieve log probabilities.

        Returns:
            the action sampled from our policy from observation obs
            the log probability of this action (this will be used for SAC)
        """
        # obs is our input observation.
        # We feed it to our actor neural network, which will output an action.

        # Let us feed it to our MLP:
        net_out = self.net(obs)
        # Now, the means of our multivariate gaussian (i.e., Normal law) are:
        mu = self.mu_layer(net_out)
        # And the corresponding standard deviations are:
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # We can now sample our action in the resulting multivariate gaussian (Normal) distribution:
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu  # at test time, our action is deterministic (it is just the means)
        else:
            pi_action = pi_distribution.rsample()  # during training, it is sampled in the multivariate gaussian
        # We retrieve the log probabilities of our multivariate gaussian as they will be useful for SAC:
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # (the next line is a correction formula for TanH squashing, present in the Spinup implementation of SAC)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        # And we squash our action within the action space:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        # Finally, we remove the batch dimension:
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    # Now, the only method that all participants are required to implement is act()
    # act() is the interface for TMRL to use your ActorModule as the policy it tests in TrackMania.
    # For the evaluation, the "test" argument will be set to True.
    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        # Since we have already implemented our policy in the form of a neural network,
        # act() is now pretty straightforward.
        # We don't need to compute the log probabilities here (they will be for our SAC training algorithm).
        # Also note that, when using TorchActorModule, TMRL calls act() in a torch.no_grad() context.
        # Thus, you don't need to use "with torch.no_grad()" here.
        # But let us do it anyway to be extra sure, for the people using ActorModule instead of TorchActorModule.
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()


# The critic module for SAC is now super straightforward:
class RNNNetQFunction(nn.Module):
    """
    Critic module for SAC.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = RNNNet(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        """
        Estimates the action-value of the (obs, act) state-action pair.

        In RL theory, the action-value is the expected sum of (gamma-discounted) future rewards
        when observing obs, taking action act, and following the current policy ever after.

        Args:
            obs: current observation
            act: tried next action

        Returns:
            The action-value of act in situation obs, as estimated by our critic network
        """
        # Since q_net is True, we append our action act to our observation obs.
        # Note that obs is a tuple of batched tensors: respectively the history of 4 images, speed, etc.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


# Finally, let us merge this together into an actor-critic torch.nn.module for training.
# Classically, we use one actor and two parallel critics to alleviate the overestimation bias.
class RNNNetActorCritic(nn.Module):
    """
    Actor-critic module for the SAC algorithm.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor):
        self.actor = MyActorModule(observation_space, action_space)
        # Value networks (critics):
        self.q1 = RNNNetQFunction(observation_space, action_space)
        self.q2 = RNNNetQFunction(observation_space, action_space)


# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam


# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinup implementation.

class SACTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC in this tutorial).

    Custom TrainingAgents implement two methods: train(batch) and get_actor().
    The train method performs a training step.
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.

    Your implementation must also pass three required arguments to the superclass:

    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,  # Gymnasium observation space (required argument here for your convenience)
                 action_space=None,  # Gymnasium action space (required argument here for your convenience)
                 device=None,  # Device our TrainingAgent should use for training (required argument)
                 model_cls=RNNNetActorCritic,  # An actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # Discount factor
                 polyak=0.995,  # Exponential averaging factor for the target critic
                 alpha=0.2,  # Value of the entropy coefficient
                 lr_actor=1e-3,  # Learning rate for the actor
                 lr_critic=1e-3):  # Learning rate for the critic

        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        # custom stuff:
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit

        Note that we will ignore the t signal for a clever reason.
        When the episode is truncated due to a time limit, we do not want our model to believe that o2 is terminal.
        Instead, we will make our model believe that the episode would normally have continued after this transition.
        Because we use a discount factor, this does not result in an infinite value.
        In our situation, the discount factor is what incentivizes the AI to run as fast as possible!

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        # We sample an action in the current policy and retrieve its corresponding log probability:
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        # We also compute our action-value estimates for the current transition:
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Now we compute our value target, for which we need to detach from gradients computation:
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        # This gives us our critic loss, as the difference between the target and the estimate:
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # We can now take an optimization step to train our critics in the opposite direction of this loss' gradient:
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # For the policy optimization step, we detach our critics from the gradient computation graph:
        for p in self.q_params:
            p.requires_grad = False

        # We use the critics to estimate the value of the action we have sampled in the current policy:
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Our policy loss is now the opposite of this value estimate, augmented with the entropy of the current policy:
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        # Now we can train our policy in the opposite direction of this loss' gradient:
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # We attach the critics back into the gradient computation graph:
        for p in self.q_params:
            p.requires_grad = True

        # Finally, we update our target model with a slowly moving exponential average:
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # TMRL enables us to log training metrics to wandb:
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict


training_agent_cls = partial(SACTrainingAgent,
                             model_cls=RNNNetActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=0.000005,
                             lr_critic=0.00003)


# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)


# =====================================================================
# RUN YOUR TRAINING PIPELINE
# =====================================================================
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run_with_wandb(entity=wandb_entity,
                                  project=wandb_project,
                                  run_id=wandb_run_id)
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=MyActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run()
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)
