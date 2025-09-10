import os, time, sys
import threading
import random

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

from a2c_ppo_acktr.config import FORMAT_SUFFIX, ENV_NAME, MODEL_NAME, MODE_LIST, DEBUG_VERBOSE
from a2c_ppo_acktr.model import ShareWorker
from a2c_ppo_acktr.modeltool import Buffer, LocalBuffer
from a2c_ppo_acktr.game.envs import make_env
from a2c_ppo_acktr.utils import pkill, get_network_iface, model_to_cpu

#=========================
class MasterRPC:
    '''
    Class of RPC in master proc.
    '''
    def __init__(self, env_num, env_ids):
        assert len(env_ids) == env_num and env_num > 0
        self.num_envs = env_num
        
        proc_env_relation = {}
        for proc_id, task_id in enumerate(env_ids):
            proc_env_relation[proc_id] = task_id
        self.proc_env_relation = proc_env_relation
        
    
    def make_vec_envs(self, env_ids, debug_log_info):
        self.env_rrefs = _make_vec_envs(env_ids, debug_log_info)
        
        self._get_env_info(self.env_rrefs[0])
        return self
    
    # def make_worker_models_env_remote(self, num):
    #     model_name = MODEL_NAME + FORMAT_SUFFIX
    #     model_rrefs = []
    #     for proc_id in range(num):
    #         print(model_name.format(proc_id))
    #         rref = remote(model_name.format(proc_id), ShareWorker, args=(None,  self.env_info))
    #         model_rrefs.append(rref)
    #     self.model_rrefs = model_rrefs
    #     print('Creating shared_models succeeds!')
    #     return self

    def make_worker_models_env_local(self):
        #create model in env local for more speed run
        env_name = ENV_NAME + FORMAT_SUFFIX
        model_rrefs = []
        for proc_id in range(self.num_envs):
            # print(model_name.format(proc_id))
            rref = remote(env_name.format(proc_id), ShareWorker, args=(None,  self.env_info))
            model_rrefs.append(rref)
        self.model_rrefs = model_rrefs
        print('Creating shared_models succeeds!')
        return self
    
    # def set_model_remote(self, shared_buffer):
    #     worker_model_rrefs = self.model_rrefs
    #     env_rrefs = self.env_rrefs
    #     model_num = len(worker_model_rrefs)
    #     for proc_id, e_rref in enumerate(env_rrefs):
    #         i = proc_id
    #         model_rref = worker_model_rrefs[i % model_num]
    #         worker_rref = remote(e_rref.owner(), RPCWokerLocal, args=(model_rref, RRef(shared_buffer), self.env_info, proc_id))
    #         rpc_sync(e_rref.owner(), set_model_rpc_local, args=(e_rref, worker_rref))

    def set_model_env_local(self, shared_buffer):
        worker_model_rrefs = self.model_rrefs
        env_rrefs = self.env_rrefs
        for proc_id, e_rref in enumerate(env_rrefs):
            model_rref = worker_model_rrefs[proc_id]
            worker_rref = remote(e_rref.owner(), RPCWokerLocal, args=(model_rref, RRef(shared_buffer), self.env_info, proc_id))
            rpc_sync(e_rref.owner(), set_model_rpc_local, args=(e_rref, worker_rref))
    
    def envs_rollout(self):
        self.rollout = RPCRolloutThread(self)
        self.rollout.start()
        # for e_rref in self.env_rrefs:
        #     rpc_async(e_rref.owner(), rollout_rpc_local, args=(e_rref,))
        
    def update_weight(self, net_state_dict):
        net_state_dict = dict(net_state_dict)
        worker_model_rrefs = self.model_rrefs
        for rref in worker_model_rrefs:
            rpc_sync(rref.owner(), ShareWorker.update_model_rpc, args=(rref, model_to_cpu(net_state_dict)))
        print('Master update rpc succeeds!')
            #{'a':1,'b':torch.ones(5)}
    
    def _get_env_info(self, env_rref):
        # env_rref = self.env_rrefs[0]
        info = rpc_sync(env_rref.owner(), get_env_info_rpc_local, args=(env_rref,))
        self.observation_space, self.action_space, self.value_observation_space = info
        self.env_info = info
            
    def env_close(self):
        for rref in self.env_rrefs:
            rpc_sync(rref.owner(), env_close_rpc_local, args=(rref,))
               
class RPCWokerLocal:
    '''
    Class of RPC in worker proc. The function of the class is same as ShareWorker class. It is a rpc local implementation class.
    '''
    def __init__(self, model_rref, buffer_rref, env_info, proc_id):
        self.set_model(model_rref)
        self.buffer_rref = buffer_rref
        self.local_buf = LocalBuffer(*env_info)
        
        self.proc_id = proc_id
        self.obs_dim, self.act_space, self.max_steps = env_info
        self.step_cou = 0
        
        self.thread = RPCLocalSender(self)
        self.thread.start()
    
    def set_model(self, model_rref):
        local_name = rpc.get_worker_info().name
        rref_name = model_rref.owner().name
        if local_name == rref_name:#local
            self.model_rref = None
            self.model = model_rref.local_value()
        else:
            self.model_rref = model_rref
            self.model = None
        
    def get_act(self, state_act_pair):
        state, expert, mask, step_cnt, proc_id = state_act_pair
        
        h0_c0 = self.local_buf.get_lstm_h0(step_cnt)
        self.local_buf.save_state_tuple(state, expert, mask, step_cnt)
        
        state_act_pair = ((state, h0_c0), expert, mask, step_cnt, proc_id)
        if self.model_rref:
            act, expert, h0_c0 = rpc_sync(self.model_rref.owner(), ShareWorker.get_act_rpc, args=(self.model_rref, state_act_pair))
        else:
            act, expert, h0_c0 = self.model.get_act(state_act_pair)
        
        self.local_buf.set_lstm_h0(h0_c0)
        #The reason of updating step_cou inthe last is that send states after putting states to buf
        #refering RPCLocalSender.run 'if  self.inner_step_cou + self.send_seq <=  self.rpc_model.step_cou'
        #change to sending states to trainer after simulating end
        # self.step_cou = step_cnt + 1 
        return (act, expert)
        
    def log_result(self, proc_id, env_id, res):
        rpc_sync(self.buffer_rref.owner(), Buffer.log_result_all_rpc, args=(self.buffer_rref, proc_id, env_id, res))     
    # def save_state_tuple_to_trainer(self, proc_id):
    #     states, acts, masks = self.local_buf.get_tuples()
    #     rpc_sync(self.buffer_rref.owner(), Buffer.save_buffer_rpc, args=(self.buffer_rref, proc_id, states, acts, masks))
        
    def save_state_tuple_segment(self, step_info, states, acts, masks):
        rpc_sync(self.buffer_rref.owner(), Buffer.save_buffer_rpc_segment, args=(self.buffer_rref, self.proc_id, step_info, states, acts, masks))
        
    def wait_finish_sending_state(self):
        self.step_cou = self.max_steps
        while not self.thread.isdone():
            time.sleep(0.1)
        self.thread.reset()
        
    
#=========================
'''
Basic function of torch.rpc.
'''
def init(rank, mode, world_size, IP='localhost', port='29513', tun=False, rpc_start_index=0):
    assert mode in MODE_LIST
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = str(port)
    
    # proxy = 'http://Naii:<passwd>@59.110.223.227:29510/'
    # os.environ['http_proxy'] = proxy 
    # os.environ['https_proxy'] = proxy
    
    NIC_iface = get_network_iface(tun)
    assert NIC_iface is not None
    #ref: https://github.com/pytorch/pytorch/issues/45196
    os.environ['GLOO_SOCKET_IFNAME'] = NIC_iface
    os.environ["TP_SOCKET_IFNAME"] = NIC_iface
    
    mode_name_str = mode + FORMAT_SUFFIX
    proc_id, rpc_rank = rank, rank + rpc_start_index
    print('Mode %s-%s' % (mode, proc_id))
    if mode == ENV_NAME:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=64,rpc_timeout=600)
    rpc.init_rpc(mode_name_str.format(proc_id),  rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    if mode == ENV_NAME:
        os.environ.pop("CUDA_VISIBLE_DEVICES")   
    
def _make_vec_envs(env_ids, debug_log_info):
    env_name = ENV_NAME + FORMAT_SUFFIX
    def _rpc_create_env(proc_id,  env_id):
        env_rpc_name = env_name.format(proc_id)
        print(env_rpc_name)
        env_rref = rpc_sync(env_rpc_name, make_env_rpc_local, args=(proc_id, debug_log_info))
        return env_rref
    
    env_rrefs = []
    for proc_id, env_id in enumerate(env_ids):
        env_rref = _rpc_create_env(proc_id,  env_id)
        env_rrefs.append(env_rref)
    return env_rrefs
    
def close():
    pass

def wait_util_shutdown():
    rpc.shutdown()
    
#=========================
'''
Local worker helper functions in worker for helping implement rpc sync/async in local proc. Just like rl_my2-script.py _call_method.
'''
def make_env_rpc_local(proc_id, debug_log_info):
    env_func = make_env(proc_id, debug_log_info)
    # time.sleep()
    env = env_func()
    return RRef(env)

def env_close_rpc_local(env_rref):
    env = env_rref.local_value()
    env.close()
    pkill()

def get_env_info_rpc_local(env_rref):
    env = env_rref.local_value()
    return [env.obs_dim, env.action_cnt, env.max_steps]

def set_model_rpc_local(env_rref, worker_model_rref):
    env = env_rref.local_value()
    worker_model = worker_model_rref.local_value()
    env.set_model(worker_model)
    
def rollout_rpc_local(env_rref, env_info):
    env = env_rref.local_value()
    res = env.reset_rollout(1, env_info)
    return res
    
#=========================
'''
Helper function and class
'''
class RPCLocalSender(threading.Thread):
    '''
    A thread implementation of RPCWokerLocal for sending state in segments to trainer.
    '''
    def __init__(self, rpc_local_model, mode='train'):
        super().__init__(daemon=True)
        self.rpc_model = rpc_local_model
        self.send_seq = 128 #send block
        
        self.inner_step_cou = 0
        self.inner_step_end = self.inner_step_cou + self.send_seq
        
        #the send_seq must can be devidied by max_steps 
        assert self.rpc_model.max_steps % self.send_seq == 0
        
    def run(self):
        while(1):
            if self.inner_step_end <= self.rpc_model.step_cou:
                # print(self.inner_step_cou, self.inner_step_end,  self.rpc_model.step_cou)
                step_info = (self.inner_step_cou, self.send_seq)
                start, end = self.inner_step_cou, self.inner_step_end
                
                states, acts, masks = self.rpc_model.local_buf.get_tuples_index(start, end)
                self.rpc_model.save_state_tuple_segment(step_info, states, acts, masks)
                
                self.inner_step_cou = self.inner_step_end
                self.inner_step_end = self.inner_step_cou + self.send_seq
                
                time.sleep(0.2) #send interval
            else:
                time.sleep(0.8)# wait simulating end
                #state interval is 10ms, so the total time is send_seq * 10ms =160
                #but 20ms is sampling time 
    
    def reset(self):
        #!!! Must set rpc_model.step_cou first, otherwise func run will send leakily
        self.rpc_model.step_cou = 0
        self.inner_step_cou = 0
        self.inner_step_end = self.inner_step_cou + self.send_seq
        
    def isdone(self):
        if self.inner_step_cou >= self.rpc_model.max_steps:
            done = True
        else:
            done = False
        return done
                
                               
class RPCRolloutThread(threading.Thread):
    def __init__(self, rpc_master_model):
        super().__init__(daemon=True)
        self.rpc_handler = rpc_master_model
    
    # def _generate_random_task_ids(self):
        
    
    def run(self):
        #0.2s is min interval between mahimahi startup
        self._init_create_task()
        self._poll()
                     
    def assign_env(self, proc_id):
        env_id = random.randint(0, 25)
        return env_id
    
    def _init_create_task(self):
        self.futs = [None] * self.rpc_handler.num_envs
        env_rrefs = self.rpc_handler.env_rrefs
        for proc_id, e_rref in enumerate(env_rrefs):
            env_info = self.assign_env(proc_id)
            self.futs[proc_id] = rpc_async(e_rref.owner(), rollout_rpc_local, args=(e_rref, env_info, ))
            time.sleep(0.3)
            
    def _poll(self):
        env_rrefs = self.rpc_handler.env_rrefs
        tic = time.time()
        while 1:
            for proc_id in range(self.rpc_handler.num_envs):
                done_res = self.futs[proc_id].done()
                if done_res:
                    try:
                        res = self.futs[proc_id].wait()
                    except:
                        print(done_res)
                        raise Exception('wait error')
                    # if DEBUG_VERBOSE:
                    #     print('proc_id: %s| content: %s' %(proc_id, res))
                    # if res is not None:
                    #     raise Exception('rpc assgin error: %s' % (res))
                    
                    env_info = self.assign_env(proc_id)
                    toc = time.time()
                    delta = 0.2 - (toc - tic)
                    if delta > 0:
                        time.sleep(delta)
                        
                    self.futs[proc_id] = rpc_async(env_rrefs[proc_id].owner(), rollout_rpc_local, args=(env_rrefs[proc_id], env_info, ))
                    tic = time.time()
                    # sys.stderr.write('%s and it was assigned new task: %s \n' % (res, env_info))
            time.sleep(0.2)
        
        
        
        