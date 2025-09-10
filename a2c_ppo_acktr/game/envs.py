#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import gym
import numpy as np
import torch
# from gym.spaces.box import Box

# from .baselines_com.atari_wrappers import WarpFrame
from .baselines_com.vec_env.vec_env import VecEnvWrapper
from .baselines_com.vec_env.shmem_vec_my import ShmemMy
from .baselines_com.vec_env.dummy_vec_my import DummyMy

from .indigo_env.dagger.worker import create_env
from .indigo_env.env.environment import Environment

def make_env(proc_id, debug_path_info):#baseline package
    def _thunk():
        # mm_cmd, best_cwnd = create_env(env_id)
        env = Environment(proc_id, debug_path_info)        
        return env

    return _thunk


def make_vec_envs(task_ids,
                  debug_path_info):
    proc_env_relation = {}
    envs = []
    for proc_id, task_id in enumerate(task_ids):
        proc_env_relation[proc_id] = task_id
        envs.append(make_env(proc_id, task_id, debug_path_info))
        
    envs = ShmemMy(envs, proc_env_relation, context='fork')

    # if gamma is None:
    #     envs = VecNormalize(envs, ret=False)
    # else:
    #     envs = VecNormalize(envs, gamma=gamma)
        
    # if device is None:
    #     device = torch.device("cpu")
    # envs = VecPyTorch(envs, device)

    # if num_frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 1, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device=None):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        # TODO: Fix data types

    # def reset(self):
    #     obs = self.venv.reset()
    #     obs = torch.from_numpy(obs).float().to(self.device)
    #     return obs

#     def step_async(self, actions):
# #        if isinstance(actions, torch.LongTensor):
# #            # Squeeze the dimension for discrete actions
# #            actions = actions.squeeze(1)
#         actions = actions.cpu().numpy()
#         self.venv.step_async(actions)

#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         return obs, reward, done, info
    


# class VecNormalize(VecEnvWrapper):
#     def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
#         VecEnvWrapper.__init__(self, venv)
#         self.training = True
        
#         from .baselines_com.vec_env.running_mean_std import RunningMeanStd
#         self.ob_rms = RunningMeanStd(shape=self.value_observation_space.shape) if ob else None
#         self.ret_rms = RunningMeanStd(shape=()) if ret else None
#         self.clipob = clipob
#         self.cliprew = cliprew
#         self.ret = np.zeros(self.num_envs)
#         self.gamma = gamma
#         self.epsilon = epsilon
    
#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.ret = self.ret * self.gamma + rews
#         if self.ret_rms:
#             if self.training:
#                 self.ret_rms.update(self.ret)
#             rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
#         self.ret[news] = 0.
#         return obs, rews, news, infos
        
#     def reset(self):
#         self.ret = np.zeros(self.num_envs)
#         obs = self.venv.reset()
#         return obs
    
    
#     def train(self):
#         self.training = True

#     def eval(self):
#         self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
# class VecPyTorchFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack, device=None):
#         self.venv = venv
#         self.nstack = nstack

#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]

#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)

#         if device is None:
#             device = torch.device('cpu')
#         self.stacked_obs = torch.zeros((venv.num_envs, ) +
#                                        low.shape).to(device)

#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, :-self.shape_dim0] = \
#             self.stacked_obs[:, self.shape_dim0:]
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs, rews, news, infos

#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs

#     def close(self):
        # self.venv.close()

        
