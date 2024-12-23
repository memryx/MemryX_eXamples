# Deep Reinforcement Learning with Mario

This example uses the MemryX Accelerators (MXA) to run a deep reinforcement learning agent which plays Super Mario Bros! This guide will demonstrate how to train a Double Deep Q Network (DDQN) agent, compile it onto the MXA, and let it play.

<p align=center>
    <img src="assets/mario.gif" alt="Mario Playing">
    <p align=center>The trained DDQN Mario agent in action.</p>
</p>


## Overview

| **Property**   | **Details**                                                                              |
| -------------- | ---------------------------------------------------------------------------------------- |
| **Model**      | [Double Deep Q Network](https://arxiv.org/pdf/1509.06461)                                |
| **Model Type** | Reinforcement Learning                                                                   |
| **Framework**  | PyTorch                                                                                  |
| **Input**      | Game state representation                                                                |
| **Output**     | Action preference                                                                        |
| **OS**         | Linux                                                                                    |
| **License**    | [MIT](LICENSE.md) |

## Environment Setup

This project is very sensitive to package versions. We recommend using `python==3.10.0` along with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 🔗 for the smoothest experience. The `gym_super_mario_bros` packages gives us the `gymnasium` environment, and we will use `torch` to define our own agent. The following commands are the exact steps used to setup this project environment:

```bash
conda create -n mario-rl python==3.10.0
pip install --extra-index-url https://developer.memryx.com/pip memryx
pip install gym-super-mario-bros==7.4.0 tensordict==0.3.0 torchrl==0.3.0 torchinfo==1.8 toml==0.10.2
```
Depending on your system, you may also need to install some additional system libraries to enable `human` rendering of the game environment.

### References
- [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros/tree/master) 🔗
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium) 🔗
- [mario-rl-tutorial.py](https://github.com/pytorch/tutorials/blob/main/intermediate_source/mario_rl_tutorial.py) 🔗

## Project Structure

| **File/Folder** | **Description**                                                   |
| --------------- | ----------------------------------------------------------------- |
| `agent.py`      | Defines our DDQN Mario agent.                                     |
| `env.py`        | Defines our training environment.                                 |
| `train.py`      | Script to train the agent using the specified configuration file. |
| `play.py`       | Script to run the agent for evaluation (or for fun!).             |
| `models.py`     | Defines the CNNs used by the agent.                               |
| `logger.py`     | Defines a simple logger used during training.                     |
| `utils.py`      | Defines helpers.                                                  |
| `cfg/`          | Contains configuration files used for training (and playing)      |

**Note: Run the scripts from the `src/` directory.**

## Download Pre-trained Example

We have provided one example agent for you to get started with. Run the commands below to download the save directory which contains the model checkpoint, compiled DFP, as well as our training configurations and logs. This agent completes the `SuperMarioBros-1-1-v2` level when used with the `--deterministic` flag of `play.py`.

```bash
cd src/runs
wget https://developer.memryx.com/example_files/mario_rl.zip
unzip mario_rl.zip -d example
cd ..
python play.py --ckpt runs/example/mario_net_final.ckpt --deterministic
```
This agent was trained for 70,000 games using three different phases.

## Training an Agent

The `train.py` script can be used to train an agent. The only required argument is `--cfg` which must be a `toml` file specifying configuration for the agent, the environment, and the training run. Refer to `cfg/default.toml` for an example you can use as a template. Run `python train.py --help` to see all arguments.

```bash
python train.py --cfg cfg/custom.toml
```
From our experiments, the training process is likely CPU bottle-necked due to the game simulation. Training on CPU vs GPU is not very different provided the model and batch size are small enough.

### Configuration

The configuration file has three tables corresponding to the agent, environment, and training run. All the parameters have standard names, but you may refer to the source code in `agent.py`, `env.py`, or `train.py` to see exactly how they are used. We recommend using `custom.toml` for all your own runs. A copy of the config file is saved for every training run (even when resuming).

| **Parameter**              | **Description**                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `model_name`               | Name of the model function or class in `models.py`.                                       |
| `lr`                       | Learning rate for the agent's optimizer (Adam).                                           |
| `gamma`                    | Discount factor for future rewards.                                                       |
| `batch_size`               | Number of experiences sampled per training step.                                          |
| `initial_exploration_rate` | Initial probability of taking a random action.                                            |
| `exploration_rate_decay`   | Factor by which the exploration rate is decayed after each step.                          |
| `exploration_rate_min`     | Minimum exploration rate.                                                                 |
| `burnin`                   | Number of steps before training starts.                                                   |
| `learn_every`              | Frequency of training steps.                                                              |
| `sync_every`               | Frequency of syncing the target network with the online network.                          |
| `name`                     | Name of the game environment. See [here][envs] 🔗.                                         |
| `render_mode`              | Mode for rendering the game environment.                                                  |
| `shape`                    | Shape of the input frames.                                                                |
| `skip`                     | Number of frames to skip between actions.                                                 |
| `moveset`                  | Name of action set or list of possible actions the agent can take. See [here][actions] 🔗. |
| `episodes`                 | Number of episodes to train the agent.                                                    |
| `print_every`              | Frequency of printing training progress.                                                  |
| `save_every`               | Frequency of saving the model checkpoint.                                                 |

[envs]: https://github.com/Kautenja/gym-super-mario-bros?tab=readme-ov-file#environments
[actions]: https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py

Note: 
- Use `exploration_rate` instead of `initial_exploration_rate` if you want to resume with a different exploration rate.
- The `[env]` table and `model_name` should not be changed when resuming a run.

### Custom Models

A couple of simple CNNs have been defined in `models.py` as examples. You may define your own custom models in this file as functions (like we have) or classes, but in either case the model must accept and use two parameters - `in_channels` and `num_actions`. To use your custom model, specify the function or class name exactly as a string in the config file you use for training.

### Resuming Training

While training it is likely you'll want to stop and resume training for whatever reason. Stop training at any time with a `KeyboardInterrupt` in the terminal, this will save the current version of the model as `mario_net_final.ckpt` before exiting. Then you may make modifications to your config file (anything except `model_name` and the `[env]` table can be changed). Then to resume training from where you left off run:

```bash
python train.py --cfg cfg/{cfg_file}.toml --resume runs/{save_dir}/mario_net_final.ckpt
```

This loads the specified checkpoint, overrides any parameters that changed in the config file, and resumes training. All logs and plots saved in `save_dir` will also continue from where they left off.

## Evaluating an Agent

Once you have a trained agent, you can watch it play using `play.py`. The only required argument is `--ckpt` which is the path to the model checkpoint you want to run. For a full list of arguments, run `python play.py --help`. We have provided one trained agent that you may use as an example. To run it on the MemryX Accelerator:

```bash
python play.py --ckpt runs/example/mario_net_final.ckpt --mxa
```

This should open up a new window where you can see this agent playing the first level of Super Mario Bros. `KeyboardInterrupt` to stop playing. You can see the details of how we trained our example agent in `runs/example/`.

### Tips & Tricks
- `--mxa`: Tries to load a pre-compiled dfp named `mario_net.dfp` from `save_dir`. If that doesn't exist, it exports the model to onnx and compiles it into a DFP during runtime. If the flag is not specified, then the model is run on cuda if available, else cpu.
- `--deterministic`: Disables exploration for the model. Ideally, a well-trained model should not need to explore at runtime, but a bit of noise can help the model complete a level occasionally instead of never.
- `--env_name`: A fun experiment is to run your model on the randomized version of the environment it was trained on!
- `--render_mode`: Rendering in `human` mode uses `pyglet` to display the game. We recommend running `play.py` from a standard terminal (outside your IDE) to make it easier to debug display dependencies. 

## Third-Party Licenses

*This project utilizes third-party software and libraries. The licenses for these dependencies are outlined below:*

- **Code Reuse**: The foundation of this implementationw was borrowed from the PyTorch tutorial [Train a Mario-playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) 🔗
  - License: [BSD-3 Clause](https://github.com/pytorch/tutorials/blob/main/LICENSE) 🔗

## Summary

This guide shows how to train a DDQN Mario agent to play Super Mario Bros and run the trained agent using the MemryX accelerator. It is quite difficult to train a good agent, so we encourage you to employ a trial and error tactic to develop your very own super-human Mario bot! Feel free to submit your own agent if you think it's good enough to replace our example!
