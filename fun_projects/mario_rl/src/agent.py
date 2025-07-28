import torch
import torch.nn as nn
import numpy as np
import onnx
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from logger import MetricLogger
from models import default_cnn
from utils import state_to_array, GREEN, YELLOW, RESET

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from memryx import NeuralCompiler, SyncAccl


class Mario:
    def __init__(
        self,
        state_dim,
        action_dim,
        env_name=None,
        model_constructor=default_cnn,
        state_dict=None,
        logger=None,
        save_dir=Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        **kwargs,
    ):
        """
        Mario is our agent.

        Args:
            state_dim (tuple): Shape of each state, used as model input shape.
            action_dim (int): Size of the action space, used as model output shape.
            env_name (str): Name of the specific Mario environment.
            model_constructor (function or class):
                A function or class constructor that returns a torch.nn.Module with identical
                function signatures to the examples in `models.py`.
            state_dict (dict): Model weights (used to resume training).
            logger (MetricLogger): (used to resume training).
            save_dir (pathlib.Path): Directory to save the model checkpoints and logs.

            **kwargs:
            lr (float): Learning rate for the optimizer. Default is 0.00025.
            gamma (float): Discount factor for future reward. Default is 0.9.
            batch_size (int): For sampling from memory. Default is 32.
            exploration_rate (float): Probability of choosing a random action. Default is 1.0.
            initial_exploration_rate (float): Initial value of exploration_rate. Default is 1.0.
            exploration_rate_decay (float): Rate at which to decay exploration_rate. Default is 0.99999975.
            exploration_rate_min (float): Minimum value of exploration_rate. Default is 0.1.
            burnin (int): Number of steps before training starts. Default is 1e4.
            learn_every (int): Number of steps between updates to Q_online. Default is 3.
            sync_every (int): Number of steps between Q_target & Q_online sync. Default is 1e4.
            state_dict (dict): State dictionary to load the model from.
        """
        if "kwargs" in kwargs:
            kwargs["kwargs"].pop("exploration_rate", None)
            kwargs.update(kwargs.pop("kwargs"))

        # Model parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_constructor = model_constructor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = kwargs.get("batch_size", 32)

        # Mario's DNNs to predict the most optimal action
        self.net = MarioNet(
            state_dim, action_dim, model_constructor, self.device, save_dir
        )

        # Load pretrained weights
        self.pretrained = False
        if state_dict:
            self.net.load_state_dict(state_dict)
            self.pretrained = True

        # Loss function and optimizer
        self.lr = kwargs.get("lr", 0.00025)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Exploration parameters
        self.initial_exploration_rate = kwargs.get("initial_exploration_rate", 1.0)
        self.exploration_rate = kwargs.get(
            "exploration_rate", self.initial_exploration_rate
        )
        self.exploration_rate_decay = kwargs.get("exploration_rate_decay", 0.99999975)
        self.exploration_rate_min = kwargs.get("exploration_rate_min", 0.1)
        self.mode = "train"

        # DDQN parameters
        self.burnin = kwargs.get("burnin", 1e4)
        self.learn_every = kwargs.get("learn_every", 3)
        self.sync_every = kwargs.get("sync_every", 1e4)
        self.gamma = 0.9
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu"))
        )
        self.env_name = env_name

        # Logging and Checkpointing
        self.save_dir = Path(save_dir)
        self.logger = logger if logger else MetricLogger(save_dir)
        self.kwargs = kwargs["kwargs"] if "kwargs" in kwargs else kwargs
        print(
            f"{GREEN}Initialized Mario with the following params:{RESET}\n{self._get_params()}\n"
        )

    @classmethod
    def from_ckpt(cls, ckpt_path, cfg_overrides=None):
        """
        Loads a checkpoint and returns a Mario agent with the same parameters.
        If agent_cfg is provided, it will override the checkpoint's configuration.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=device)

        if cfg_overrides:
            # Can't change model architecture
            cfg_overrides.pop("model_name", None)
            # Overrides the checkpoint's configuration
            if "exploration_rate" in cfg_overrides:
                ckpt["exploration_rate"] = cfg_overrides.pop("exploration_rate")
                ckpt["kwargs"].pop("exploration_rate", None)
            ckpt["kwargs"].update(cfg_overrides)

        agent = Mario(**ckpt)
        return agent

    def train(self):
        "Enables exploration for learning."
        self.mode = "train"

    def val(self):
        "Disables exploration for evaluation."
        self.mode = "val"

    # Action
    def act(self, state, mxa=False):
        """
        Given a state, choose an epsilon-greedy action and update the step count.

        Args:
            state (LazyFrame): A single observation of the current state, dimension is (state_dim).
            mxa (bool): Whether to run the model on MXA.

        Returns:
            int: An integer representing which action Mario will perform.
        """
        # Explore (random action)
        if self.mode == "train" and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # Exploit (optimal action)
        else:
            state = state_to_array(state)[np.newaxis, ...]
            if not mxa:
                state = torch.tensor(state, device=self.device)
            action_values = self.net(state, model="online", mxa=mxa)
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decrease exploration_rate over time
        if self.mode == "train":
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(
                self.exploration_rate_min, self.exploration_rate
            )

        return action_idx

    # Memory
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Args:
            state (LazyFrame): The current state.
            next_state (LazyFrame): The next state after taking the action.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode is done.
        """
        # Extract the numpy arrays from the LazyFrames
        state = state_to_array(state)
        next_state = state_to_array(next_state)

        # Construct TensorDict and add to memory
        self.memory.add(
            TensorDict(
                {
                    "state": torch.tensor(state),
                    "next_state": torch.tensor(next_state),
                    "action": torch.tensor([action]),
                    "reward": torch.tensor([reward]),
                    "done": torch.tensor([done]),
                },
                batch_size=[],
            )
        )

    def recall(self):
        """Retrieve a batch of experiences from memory."""
        # Randomly sample memory and extract TensorDict values
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # Learning
    def learn(self):
        """Double Deep Q Networks (DDQN) - https://arxiv.org/pdf/1509.06461"""
        curr_step = self.logger.curr_step

        # Sync weights
        if curr_step % self.sync_every == 0:
            self.sync_Q_target()
        # Warmup (no learning)
        if curr_step < self.burnin:
            return None, None
        # Experience (no learning)
        if curr_step % self.learn_every != 0:
            return None, None

        # Learn from experience stored in memory
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        """Returns Q-value (using Q_online) of taking action in state (batched)"""
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Returns expected future return (using Q_online) calculated as current reward
        plus the discounted future reward of taking the optimal action (from Q_target)
        in next_state (batched).
        """
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """Updates Q_online model parameters with backpropagation and returns loss."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Syncs parameters of Q_target with Q_online."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    # Utilities
    def save(self, save_path=None, print_fn=print):
        """Saves a model checkpoint."""
        model_name = self.net.__class__.__name__
        save_path = (
            save_path
            if save_path
            else (self.save_dir / f"{model_name.lower()}_{self.logger.episode}.ckpt")
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.kwargs.pop("exploration_rate", None)
        torch.save(
            dict(
                state_dict=self.net.state_dict(),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                env_name=self.env_name,
                model_constructor=self.model_constructor,
                save_dir=self.save_dir,
                exploration_rate=self.exploration_rate,
                logger=self.logger,
                kwargs=self.kwargs,
            ),
            save_path,
        )
        print_fn(
            f"{YELLOW}{model_name} saved to {save_path} at step {self.logger.curr_step}{RESET}"
        )

    def _get_params(self):
        """Returns the parameters of the agent."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "env_name": self.env_name,
            "model_constructor": self.model_constructor,
            "pretrained": self.pretrained,
            "save_dir": self.save_dir,
            "device": self.device,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "exploration_rate": self.exploration_rate,
            "initial_exploration_rate": self.initial_exploration_rate,
            "exploration_rate_decay": self.exploration_rate_decay,
            "exploration_rate_min": self.exploration_rate_min,
            "burnin": self.burnin,
            "learn_every": self.learn_every,
            "sync_every": self.sync_every,
            "gamma": self.gamma,
        }


class MarioNet(nn.Module):
    """
    A simple CNN network:
        Input -> (Conv2d + ReLU) x 3 -> Flatten -> (Dense + ReLU) x 2 -> Output
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        model_constructor,
        device="cpu",
        save_dir=Path("runs/tmp"),
    ):
        """
        Args:
            input_dim (tuple): The shape of the input tensor.
            output_dim (int): The shape of the output tensor.
            model_constructor (function or class):
                A function or class constructor that returns a torch.nn.Module with identical
                function signatures to the examples in `models.py`.
            device (str): The device to run the model on.
            save_dir (pathlib.Path): Directory to save the model checkpoints and logs.
        """
        super().__init__()
        # Used for DFP compilation
        self.input_dim = input_dim
        self.device = device
        self.save_dir = save_dir
        c, h, w = input_dim

        # Double Deep Q Networks (DDQN) with shared weights
        self.model_constructor = model_constructor
        self.online = self.model_constructor(c, output_dim)
        self.target = self.model_constructor(c, output_dim)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.to(device).float()

    def forward(self, input, model, mxa=False):
        """
        Perform a forward pass through the specified model.

        Args:
            input (torch.Tensor): A tensor representing the state.
            model (str): The model to use for the forward pass. Must be one of "online" or "target".
            mxa (bool): Whether to run the model on MXA, only applies to online model.

        Returns:
            torch.Tensor: The output tensor with dimensions (batch_size, action_dim).
        """
        if model == "online":
            if mxa:
                return self.mxa_forward(input)
            else:
                return self.online(input)
        elif model == "target":
            return self.target(input)

    def mxa_forward(self, input):
        """
        Perform a forward pass through the online model on the MXA.

        Args:
            input (torch.Tensor): A tensor representing the state.

        Returns:
            torch.Tensor: The output tensor with dimensions (batch_size, action_dim).
        """
        # Check that model has been compiled
        if not hasattr(self, "accl") or not hasattr(self, "dfp"):
            raise AttributeError(
                "Compile the model to a DFP with MarioNet.compile_dfp() before running on MXA."
            )
        out = self.accl.run(input)
        out = torch.from_numpy(out).to(self.device)
        return out

    def compile_dfp(self, model_name="mario_net", load_precompiled=True, verbose=0):
        """
        Exports the model to ONNX and then compiles it to a DFP for the MXA.
        Loads the dfp if it exists already (skipping export and compilation).

        Args:
            model_name (str): Name of the model without extensions.
            load_precompiled (bool): Whether to load a precompiled DFP if it exists.
            verbose (int): Verbosity level for the NeuralCompiler.
        """
        dfp_path = self.save_dir / f"{model_name}.dfp"

        if dfp_path.exists() and load_precompiled:
            # Skipping export and compilation
            print(f"{YELLOW}Loading precompiled DFP from {dfp_path}.{RESET}\n")
            self.dfp = str(dfp_path)
        else:
            # Export the model to ONNX
            onnx_path = self.save_dir / f"{model_name}.onnx"
            print(f"{YELLOW}Exporting model to ONNX to {onnx_path}.{RESET}")
            dummy_input = torch.randn(1, *self.input_dim).to(self.device)
            torch.onnx.export(self.online, dummy_input, onnx_path)
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model, full_check=True)
            # Compile the model to a DFP
            print(f"{YELLOW}Compiling model to DFP to {dfp_path}.{RESET}\n")
            nc = NeuralCompiler(
                models=str(onnx_path), dfp_fname=str(dfp_path), verbose=verbose
            )
            self.dfp = nc.run()

        # Create a SyncAccl object to reuse
        self.accl = SyncAccl(self.dfp)


if __name__ == "__main__":
    """Prints a summary of the MarioNet model."""
    from torchinfo import summary

    state_dim = (4, 84, 84)
    action_dim = 2

    mario = Mario(state_dim, action_dim)
    input_size = (1, *state_dim)

    print(
        "Q-Online and Q-Target have identical architectures, but the latter is frozen."
    )
    summary(mario.net.online, input_size=input_size)
