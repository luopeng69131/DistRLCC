#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import multiprocessing as mp
import torch.multiprocessing as mp
import numpy as np
import ctypes
from .vec_env import VecEnv,CloudpickleWrapper, clear_mpi_env_vars
from .util import dict_to_obs, obs_space_info, obs_to_dict
#from baselines import logger


_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}

class ShmemMy(VecEnv):
    def __init__(self, env_fns, proc_env_relation=None, context='spawn'):
        pass
#        super().__init__(env_fns, spaces,context)
        ctx = mp.get_context(context)
 
        self.proc_env_relation = proc_env_relation

        dummy = env_fns[0]()
        observation_space, action_space, value_observation_space = dummy.observation_space, dummy.action_space, dummy.value_observation_space
        dummy.close()
        del dummy
        
        VecEnv.__init__(self, len(env_fns), observation_space, action_space, value_observation_space)
        # self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        # self.obs_bufs = [
        #     {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
        #     for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            self.lock = ctx.Lock()
            for env_fn in env_fns:
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                                   args=(child_pipe, parent_pipe, wrapped_fn, self.lock))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
                # time.sleep(1)
        self.waiting_step = False
        self.viewer = None

    def init(self):
        for pipe in self.parent_pipes:
            pipe.send(('init', None))
        return [pipe.recv() for pipe in self.parent_pipes]
    
    def reset(self):
        if self.waiting_step:
#            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return [pipe.recv() for pipe in self.parent_pipes]
    
    def set_model(self, model):
        for pipe in self.parent_pipes:
            # pipe.send(('setAct', CloudpickleWrapper(model)))
            pipe.send(('setAct', model))
        model.set_lock(self.lock)
        return [pipe.recv() for pipe in self.parent_pipes]
       
    def rollout(self):
        for pipe in self.parent_pipes:
            pipe.send(('rollout', None))
        self.waiting_step = True
        
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        tputs, delays, cwnds = zip(*outs)
        return tputs, delays, cwnds
    
    def reset_rollout(self, epoch):
        for pipe in self.parent_pipes:
            pipe.send(('resetRoll', epoch))
        self.waiting_step = True
        
        # outs = [pipe.recv() for pipe in self.parent_pipes]
        # self.waiting_step = False
    

    def step_async(self, actions):
        self.waiting_step = True

    def step_wait(self):
        self.waiting_step = False


    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            try:
                pipe.send(('close', None))
            except:
                pass
        for pipe in self.parent_pipes:
            try: 
                pipe.recv()
            except:
                print('child pipe has leaved')
            pipe.close()
        for proc in self.procs:
            proc.join()


#     def _decode_obses(self, obs):
#         result = {}
#         for k in self.obs_keys:

#             bufs = [b[k] for b in self.obs_bufs]
#             o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
#             result[k] = np.array(o)
#         return dict_to_obs(result)

# def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
    # def _write_obs(maybe_dict_obs):
    #     flatdict = obs_to_dict(maybe_dict_obs)
    #     for k in keys:
    #         dst = obs_bufs[k].get_obj()
    #         dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
    #         np.copyto(dst_np, flatdict[k])
    
def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, lock):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    
    env = env_fn_wrapper.x()
    env.set_lock(lock)
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                env.reset()
                pipe.send(None)
            elif cmd == 'init':
                pipe.send(env.init())
            elif cmd == 'setAct':
                pipe.send(env.set_model(data))
                # pipe.send(env.set_model(data.x)) 
            elif cmd == 'rollout':
                pipe.send(env.rollout())
            elif cmd == 'resetRoll':
                env.reset_rollout(data)
                # pipe.send(env.reset_rollout(data))
            elif cmd == 'close':
                env.cleanup()
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt: #
        print('ShmemVecEnv worker: KeyboardInterrupt\n')
    finally:
        env.cleanup()
        
