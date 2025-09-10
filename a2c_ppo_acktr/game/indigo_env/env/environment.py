# Copyright 2018 Francis Y. Yan, Jestin Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import os
from os import path
import sys
import signal
from subprocess import Popen, call, PIPE
from .sender import Sender, SenderStateThread
from . import project_root
from helpers.helpers import get_open_udp_port
from dagger.worker import create_env
import numpy as np
import time

DEBUG = False

class Environment:
    def __init__(self, proc_id, debug_path_info):
        self.proc_id = proc_id
        self.env_id = None
        self.state_dim = Sender.state_dim
        self.obs_dim = Sender.obs_dim #nn model state
        self.action_cnt = Sender.action_cnt
        self.max_steps = Sender.max_steps
        
        self.debug_path_info  = debug_path_info
        self.verbose = debug_path_info[-1]
        
        self.debug = DEBUG
        
        # self.mahimahi_cmd = mahimahi_cmd
        # self.best_cwnd = best_cwnd
        # self.mahimahi_dict = {}
        # for i in range(25):
        #     self.mahimahi_dict[i] = create_env(i)
            
        self.retry_times = 0
        self.run_time = 0
        self.port = get_open_udp_port()

        # variables below will be filled in during setup
        self.sender = None
        self.receiver = None
        
        self.action_space = self.action_cnt
        self.observation_space = self.obs_dim
        self.value_observation_space = Sender.max_steps
        
        self.running = True
        self.sender_thread = SenderStateThread(self.proc_id, self.debug, None)
        self.sender_thread.start()
        

    # def set_lock(self, lock):
    #     self._env_lock = lock
    
    def set_model(self, model):
        self.model = model
        # self.model.init_local_var()
        # assert self._env_lock is not None
        # self.model.set_lock(self._env_lock)
        
    def reset(self, env_info):
        """Must be called before running rollout()."""
        
        self.env_id = env_info
        self.mahimahi_cmd, self.best_cwnd = create_env(self.env_id)
        self.run_time += 1
        
        self.cleanup()
        # time.sleep(0.1*self.proc_id)
        # self.port = get_open_udp_port()

        # start sender as an instance of Sender class
        self.sender = Sender(self.port, train=True, debug=self.debug, proc_id=self.proc_id,\
                             env_id=self.env_id, best_cwnd=self.best_cwnd, debug_path_info=self.debug_path_info, thread = self.sender_thread)
        self.sender.set_model(self.model)
        # self.sender.set_sample_action(self.sample_action)

        # start receiver in a subprocess
        # sys.stderr.write('Starting receiver...\n')
        receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        self.receiver_src = receiver_src
        recv_cmd = 'python %s $MAHIMAHI_BASE %s %s %s' % (receiver_src, self.port, self.env_id, int(self.debug))
        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, recv_cmd)
        # sys.stderr.write('$ %s\n' % cmd)
        self.receiver = Popen(cmd, preexec_fn=os.setsid, shell=True)

        # sender completes the handshake sent from receiver
        try:
            self.sender.reset()
            self.sender.handshake()
        except OSError:
            self.retry_times += 1
            if self.retry_times < 2:
                sys.stderr.write('Env %s retrys to reconnect %s...\n' % (self.env_id, self.retry_times))
                self.receiver = None
                self.reset()
            else:
                raise Exception('Env %s retry time exceeds limits' % (self.env_id))

    def rollout(self):
        """Run sender in env, get final reward of an episode, reset sender."""
        
        # sys.stderr.write('Obtaining an episode from environment...\n')
        self.sender.run()
        # self.sender.wavehand()
        result = (self.sender.tput, self.sender.perc_delay, self.sender.loss, self.sender.expert_acc_rate, self.sender.best_cwnd_rate, self.sender.write_log_flag, self.sender.cwnd)
        running_log = [self.sender.running_rtts, self.sender.running_cwnd, self.sender.running_delta_cwnd,  self.sender.state_delay, self.sender.state_delivery, self.sender.state_send]
        text = self.sender.send_thread._write_text
        if text:
            text = 'Env id: %s, episode: %s.\n||' % (self.env_id, self.run_time) + text
        return result, running_log, text

    def cleanup(self):
        flag = False
        if self.sender:
            flag = True
            self.sender.cleanup()
            self.sender = None
            
        if self.receiver:
            flag = True
            self.receiver.kill()
            # self.receiver = None
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
                # os.kill(self.receiver.pid, signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
            finally:
                self.receiver = None
                # pkill_cmds = 'pkill -f %s' % (self.receiver_src)
                # call(pkill_cmds, shell=True)
                
    def reset_rollout(self, max_episode, env_info):
        # assert epoch > 0
#         if max_episode is None:
#             max_episode = float('inf')
        
        episode = 0
        while episode < max_episode and self.running:
            self.reset(env_info)
            (tput, delay, loss, expert_acc, best_cwnd_rate, log_flag, cwnd), running_log, text = self.rollout()
            
            episode += 1
            # self.model.save_state_tuple_to_trainer(self.proc_id)
            self.model.wait_finish_sending_state()
            self.model.log_result(self.proc_id, self.env_id, [(tput, delay, loss, expert_acc, best_cwnd_rate, log_flag), running_log, text])
            sys.stderr.write('[proc_id: %s , env_id: %s]| tput: %.3f, delay: %.3f, cwnd: %.2f \n' % (self.proc_id, self.env_id, tput, delay, cwnd))
        
        return '[proc_id: %s , env_id: %s] is finished!' % (self.proc_id, self.env_id)
        
        
    def close(self):
        self.running = False
        self.cleanup()
                
