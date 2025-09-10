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


import time
import sys
import json
import socket
import select
import threading
from os import path
from queue import Queue, Full
import numpy as np
from . import datagram_pb2
from . import project_root 
from dagger.experts import get_best_action
from helpers.helpers import (
    curr_ts_ms, apply_op, normalize, one_hot, normalize_new_add,
    READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)

TIMEOUT = 1000  # ms
MAX_CWND = 8000
# TEST = False
# DEBUG_LOG_PATH = './runs/debug_log'
# LOG = True

def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
                  for idx, action in enumerate(action_list)}



#ref thread safey: https://blog.csdn.net/LAM1006_csdn/article/details/123727136
#ref daemon: http://t.zoukankan.com/i-honey-p-8047837.html
#ref thread class: http://c.biancheng.net/view/2603.html
class SenderStateThread(threading.Thread):
    def __init__(self, proc_id, debug, debug_path_info):
        super().__init__(daemon=True)
        
        self.state_p = Queue(1)
        self.act_q = Queue(1)
        self.mutex = threading.Lock()
        
        self.proc_id = proc_id
        # self.env_id = env_id
        
        self.step_cnt = 0
        
        self.debug = debug
        # if self.debug:
        #     self.thread_file = open(path.join(DEBUG_LOG_PATH, 'thread_time_'+str(env_id)), 'w', 1)
        
        # self.log_interval_file = open(path.join(*debug_path_info)+str(proc_id), 'w', 1)
        
        
    def run(self):
        while(1):
            # sys.stderr.write('Thread test get!\n')
            #joining get not needing lock
            state_act_pair, log_time = self.state_p.get()
            
            state, expert, mask, step_cnt, env_id = state_act_pair
            
            act = self.model.get_act(state_act_pair)
            # act = expert #self.model.get_act(state_act_pair)
            
            if self.step_cnt != step_cnt:
                # self._log_write_count += self.log_interval_file.write('env_thread id: %s |IN step_cnt err: thread- %s, now-%s \n' % (self.env_id, self.step_cnt, step_cnt))
                # self._write_text = 'env_thread id: %s,IN step_cnt err: thread- %s and now-%s \n' % (self.env_id, self.step_cnt, step_cnt) + self._write_text
                self._log_in_err += 1
                self.step_cnt = step_cnt
            self.step_cnt += 1
            
            with self.mutex:
                if self.act_q.full():
                    self.act_q.get()    
                self.act_q.put((act, self.step_cnt, log_time))
                            
    
    def set_model(self, model):
        self.model = model
    
    def close(self):
        pass
        # if self.debug:
        #     self.thread_file.close()
        # self.log_interval_file.close()
            
        
    
class Sender(object):
    # RL exposed class/static variables
    max_steps = 1024
    state_dim = 5
    action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
    action_cnt = len(action_mapping)
    obs_dim = state_dim + action_cnt

    def __init__(self, port=0, train=False, debug=True, \
                 proc_id=-1, env_id=-1, best_cwnd=None, model=None,debug_path_info=None,thread=None):
        self.train = train
        self.debug = debug
        self.env_id = env_id
        self.proc_id = proc_id
        self.port = port
        self.best_cwnd = best_cwnd
        
        if debug_path_info is None:
            self.verbose = False
        else:
            self.verbose = debug_path_info[-1]
        
        self.run_time = 0
        self.running = False
        
        # UDP socket and poller
        self.peer_addr = None
        self.handshake_first_flag = False
        self.init_conn()

        self.dummy_payload = 'x' * 1400
        if thread is None:
            self.send_thread = SenderStateThread(self.proc_id, self.debug, debug_path_info)
            self.send_thread.start()
        else:
            self.send_thread = thread
        
        
        # if self.debug:
        #     self.sampling_file = open(path.join(DEBUG_LOG_PATH, 'sampling_time_'+str(self.env_id)), 'w', 1)
        # if TEST:
        #     self.test_file = open(path.join(DEBUG_LOG_PATH, 'test_log_'+str(self.env_id)), 'w', 1)
            
            
    def set_model(self, model):
        self.model = model
        self.send_thread.set_model(model)
            
    
    def reset(self):
        '''For reusing Sender obj.'''
        self.run_time += 1
        
        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10.0
        self.step_len_ms = 10

        # state variables for RLCC
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')
        self.delay_ewma = 0
        self.send_rate_ewma = 0
        self.delivery_rate_ewma = 0
        
        self.rtts = None
        self.rttd = None
        self.wait_confirm_pkgs = {}
        self._rtt_pkgs_init_num = float('inf')
        self.confirm_loss_flag = False
        self.waiting_thread_syn_flag = False
        
        self.running_rtts = []
        self.running_cwnd = []
        self.running_delta_cwnd = []
        self.state_delay = []
        self.state_delivery = []
        self.state_send = []
        
        
        if self.train:
            self.old_cwnd = self.cwnd
            self.expert_act = None
            self.expert_acc_num = 0
            self.expert_acc_rate = 0
            self.best_cwnd_rate = 0
            self.best_cwnd_num = 0
            self.best_cwnd_thres = self.best_cwnd - 20 if self.best_cwnd >= 35 else self.best_cwnd
            
            self.expert_act_seq = self.generate_expert_act_seq()
        
        # self.send_pkgs_num = 0
        self.recv_pkgs_num = 0

        self.step_start_ms = None
        self.running = True
        
        self.prev_action = 2
        self.recv_state_act_flag = True
        
        # if self.debug:
        #     self.log_start_time = None
        
        self.step_cnt = 0
        if self.train:
            self.ts_first = None
            self.rtt_buf = []
            
            self.tput, self.perc_delay = 0, 0
        
        #clear pipe
        self.send_thread.step_cnt = 0
        with self.send_thread.mutex:
            if self.send_thread.act_q.full():
                self.send_thread.act_q.get()
        
        #err_log
        self._err_state_put = 0
        self._log_step_err = 0
        self.send_thread._log_in_err = 0
        self.send_thread._write_text = ''
        
             
# ==============================================================================
#                         run and end
# ==============================================================================                    
    def run(self):
        self.sock.setblocking(0)  # non-blocking UDP socket
        

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS

        while self.running:
            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS
            
            wait_time, wait_flag = self._retransmit_and_confirm_loss()
            events = self.poller.poll(wait_time)

            if not events and not wait_flag:  # timed out
                self.send()

            for fd, flag in events:
                # assert self.sock.fileno() == fd
                fileno = self.sock.fileno()
                if fileno != fd:
                    raise Exception('proc_id: %s, env: %s, step_cnt: %s| fileno %s, fd: %s ;' % (self.proc_id, self.env_id, self.step_cnt, fileno, fd))

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    self.recv()

                if flag & WRITE_FLAGS:
                    if self.window_is_open():
                        self.send()
                        

   
    def compute_performance(self):
        duration = curr_ts_ms() - self.ts_first
        tput = 0.008 * self.delivered / duration
        perc_delay = np.percentile(self.rtt_buf, 95)
        loss = (1 - self.recv_pkgs_num / self.seq_num)*100

        # with open(path.join(project_root.DIR, 'env', 'perf'), 'a', 0) as perf:
        #     perf.write('%.2f %d\n' % (tput, perc_delay))
        self.tput, self.perc_delay, self.loss = tput, perc_delay/2, loss
        self.expert_acc_rate = self.expert_acc_num * 100/ Sender.max_steps
        self.best_cwnd_rate = self.best_cwnd_num *100 /  Sender.max_steps
        
        if self._err_state_put > 0: 
            self.send_thread._write_text = 'env id: %s, put err: %s. ||\n' % (self.env_id, self._err_state_put) + self.send_thread._write_text
            
        if self.send_thread._log_in_err > 0:
            self.send_thread._write_text = 'env id: %s, In err: %s. ||\n' % (self.env_id, self.send_thread._log_in_err) + self.send_thread._write_text
            
        if self._log_step_err > 1:
            self.send_thread._write_text = 'thread id: %s, step_err_num: %s. ||\n' % (self.env_id, self._log_step_err) + self.send_thread._write_text
            
        if self.send_thread._write_text:
            self.write_log_flag = 1  
            # self.send_thread._write_text = 'Env id: %s, episode: %s.\n||' % (self.env_id, self.run_time) + self.send_thread._write_text
        else:
            self.write_log_flag = 0
        
        # if self.write_log_flag > 0:
        #     # self.send_thread.log_interval_file.write(self.send_thread._write_text)
                
# ==============================================================================
#                         recv and update
# ==============================================================================

    def update_state(self, ack):
        """ Update the state variables listed in __init__() """
        self.recv_pkgs_num += 1
        if self.step_cnt <= self._rtt_pkgs_init_num:
            self.next_ack = max(self.next_ack, ack.seq_num + 1)
        else:
            if ack.seq_num in self.wait_confirm_pkgs:
                self.wait_confirm_pkgs.pop(ack.seq_num)
                if self.wait_confirm_pkgs:
                    self.next_ack = next(iter(self.wait_confirm_pkgs)) + 1
                else:
                    self.next_ack = ack.seq_num + 1
        curr_time_ms = curr_ts_ms()

        # Update RTT
        rtt = float(curr_time_ms - ack.send_ts)
        self.min_rtt = min(self.min_rtt, rtt)
        
        if self.rtts is not None:
            self.rtts = 0.875 * self.rtts + 0.125 * rtt
            self.rttd = 0.75 * self.rttd + 0.25 * abs(self.rtts - rtt)
            self.rto = self.rtts + 4 * self.rttd
        else:
            self.rtts, self.rttd = rtt, 0.5 * rtt
        
        if self.train:
            if self.ts_first is None:
                self.ts_first = curr_time_ms
            self.rtt_buf.append(rtt)

        delay = rtt - self.min_rtt
        if self.delay_ewma > 0:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay
        else:
            self.delay_ewma = delay

        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) /
                         max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma > 0:
            self.delivery_rate_ewma = (
                0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)
        else:
            self.delivery_rate_ewma = delivery_rate
            

        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)

        if self.send_rate_ewma > 0:
            self.send_rate_ewma = (
                0.875 * self.send_rate_ewma + 0.125 * send_rate)
        else:
            self.send_rate_ewma = send_rate
            
        # print('2', end='')
            
    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)
        # serialized_ack = serialized_ack

        if addr != self.peer_addr:
            return

        ack = datagram_pb2.Ack()
        ack.ParseFromString(serialized_ack)
        
        # if self.debug:
        #     self.sampling_file.write('---Env: %s, ack %s, seq %s\n' % (self.env_id, ack.seq_num, self.seq_num))

        self.update_state(ack)

        if self.step_start_ms is None:
            self.step_start_ms = curr_ts_ms()

        # At each step end, feed the state:
        if curr_ts_ms() - self.step_start_ms > self.step_len_ms:  # step's end
            #update cwnd
            self.window_is_open() 
            
            basic_state = [self.delay_ewma,
                           self.delivery_rate_ewma,
                           self.send_rate_ewma,
                           self.cwnd]
            new_add_state = [self.min_rtt]
            
            # time how long it takes to get an action from the NN
            # if self.debug:
            #     start_sample = time.time()
            self.sample_action(basic_state, new_add_state)
            
            if self.train:
                self.running_rtts.append(self.rtts)
                self.running_cwnd.append(self.cwnd)
                self.running_delta_cwnd.append(self.cwnd - self.old_cwnd)
                self.state_delay.append(self.delay_ewma)
                self.state_delivery.append(self.delivery_rate_ewma)
                self.state_send.append(self.send_rate_ewma)
            
                self.old_cwnd = self.cwnd
                if self.cwnd >= self.best_cwnd_thres:
                    self.best_cwnd_num += 1
        
            self.delay_ewma = 0
            self.delivery_rate_ewma = 0
            self.send_rate_ewma = 0
            self.confirm_loss_flag = False

            self.step_start_ms = curr_ts_ms()
            
            self.step_cnt += 1
            if self.train:
                # self.state_act_pairs.put((state,action))
                if self.step_cnt >= Sender.max_steps:
                    self.running = False
                    self.compute_performance()

    def sample_action(self, state, new_add_state):
        """Get the act. async call."""
        
        cwnd = state[3]
        confirm_loss_flag = self.confirm_loss_flag
        if self.train:
            expert_action = get_best_action(self.action_mapping, cwnd, self.best_cwnd)
        else:
            expert_action = None
            
        mask = [1] * self.action_cnt
        if confirm_loss_flag is True:
            mask[3:] = [0, 0]
            if self.train:
                expert_action = min(2, expert_action)
        self.expert_act = expert_action
        
        # if self.step_cnt > 0:
        #     state = expert_action - self.last_state_test
        #     if state < 0:
        #         state = state + 5
        # else:
        #     state = expert_action
        # self.last_state_test = state
        # aug_state = one_hot(state, self.action_cnt) + [0] * 5
        
        # norm_state = normalize(state)
        # new_add_state = normalize_new_add(new_add_state)
        #! normalize in model
        aug_state = state + new_add_state + one_hot(self.prev_action, self.action_cnt)
       
        send_state = (aug_state, expert_action, mask, self.step_cnt, self.proc_id)
        send_state = (send_state,  time.time())
            
        
        try:
            self.send_thread.state_p.put_nowait(send_state)
            self.recv_state_act_flag = False  #set send state flag
        except Full:
            self._err_state_put += 1
                
        return None
        # if TEST:
        #     if self.step_cnt % 30 == 0:
        #         self.test_file.write('id: %s | RTTs: %.2f | RTTd: %.2f | next_ack: %s | seq: %s \n'  \
        #                                 % (self.env_id,  self.rtts, self.rttd, self.next_ack, self.seq_num))
        


# ==============================================================================
#                      send action
# ==============================================================================
    def send(self):
        data = datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = curr_ts_ms()
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload
        
        self.log_send_pkgs(data)

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)
        # try:
        #     self.sock.sendto(serialized_data, self.peer_addr)
        # except:
        #     raise Exception('proc_id: %s, env: %s, step_cnt: %s| fileno:%s, running: %s' % (self.proc_id, self.env_id, self.step_cnt, self.sock.fileno(),self.running))
        
        self.seq_num += 1
        self.sent_bytes += len(serialized_data)
        
    def log_send_pkgs(self, data):
        #first send
        if self.step_cnt > self._rtt_pkgs_init_num:
            rto = min(self.rto, TIMEOUT)
            resend_time = data.send_ts + rto
            self.wait_confirm_pkgs[data.seq_num] = [resend_time, data]

                                                 
    def take_action(self, action_idx):
        old_cwnd = self.cwnd
        op, val = self.action_mapping[action_idx]

        self.cwnd = apply_op(op, self.cwnd, val)
        self.cwnd = max(2.0, self.cwnd)
        self.cwnd = min(MAX_CWND, self.cwnd)
        self.prev_action = action_idx
        
        

    def window_is_open(self):
        if self.recv_state_act_flag is False:
            # avoid to repeat going critical region when act had been take after send state
            action = None

            with self.send_thread.mutex:
                if self.send_thread.act_q.full():
                    action = self.send_thread.act_q.get()

            if action is not None:
                act_expert, step_thread, log_time = action
                if step_thread == self.step_cnt:
                    act_idx, expert = act_expert
                    if self.train:
                        if act_idx == expert:#cal expert acc
                            self.expert_acc_num += 1
                        use_act = expert if self.expert_act_seq[self.step_cnt] else act_idx
                    else:
                        use_act = act_idx
                    self.take_action(use_act) #model-policy
                    self.recv_state_act_flag = True
                else:
                    if self._log_step_err < 1:
                        self.send_thread._write_text = 'thread id: %s, OUT stepinfo: thread-%s, now-%s, time: %.2f ms. ||\n' % (self.env_id, step_thread, self.step_cnt, (time.time() - log_time) * 1000) + self.send_thread._write_text
                    self._log_step_err += 1
        return self.seq_num - self.next_ack < self.cwnd
    
        
    def _retransmit_and_confirm_loss(self):
        min_wait_time = TIMEOUT
        wait_flag = False 
        if self.step_cnt > self._rtt_pkgs_init_num:
            now_time = curr_ts_ms()
            remove_seqs = []
            for seq_num in iter(self.wait_confirm_pkgs):
                resend_time, _ = self.wait_confirm_pkgs[seq_num]
                if now_time < resend_time:
                    min_wait_time = min(resend_time - now_time, min_wait_time)
                    #wait_flag setting true means that waitting pkgs needing to be confirmed 
                    wait_flag = True
                    break
                else:
                    remove_seqs.append(seq_num)
                    self.next_ack = seq_num + 1
                    self.confirm_loss_flag = True
            #although using extra loop to remove overtime seqs, 
            #the number of overtime pkgs is little, thus making no obvious time consuming
            #And when dict in loop, the dict can not be add or del
            for seq_remove in remove_seqs:
                self.wait_confirm_pkgs.pop(seq_remove)
            
        return min_wait_time, wait_flag

        
# ==============================================================================
#                      handshaeke and wavehand
# ==============================================================================    
    def init_conn(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.sock.bind(('0.0.0.0', self.port))
        if self.verbose:
            sys.stderr.write('[proc %s | sender %s] Listening on port %s\n' %
                             (self.proc_id, self.env_id, self.sock.getsockname()[1]))

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)
        
        self.sock.setblocking(0)
        # try:
        #     self.sock.recv()
        # except:
        #     pass
        
    def cleanup(self):
        # if self.debug and self.sampling_file:
        #     self.sampling_file.close()
        self.sock.close()
        self.send_thread.close()
        self.send_thread = None
        if self.running:
            raise Exception('Ilegal close |proc_id: %s, env: %s, step_cnt: %s, running: %s' % (self.proc_id, self.env_id, self.step_cnt, self.running))
      
    def handshake(self):
        if not self.handshake_first_flag:
            self._handshake_first()
            self.handshake_first_flag = True
        else:
            self._handshake_others()

    
    def _handshake_others(self):
        """Handshake with peer receiver. Must be called before run()."""
        
        self.sock.setblocking(0)  # non-blocking UDP socket
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        
        self._retransmit('Hello from sender','Hello from receiver')
        # sys.stderr.write('[sender %s] Handshake success! '
        #                 'Receiver\'s address is %s:%s\n' % (self.env_id, *self.peer_addr))
        
    def _handshake_first(self):
        """Handshake with peer receiver. Must be called before run()."""

        self.sock.setblocking(1) #blocking UDP socket
        wait_msg = 'Hello from receiver'
        wait_msg_encode = wait_msg.encode()
        while True:
            msg, addr = self.sock.recvfrom(1600)
            # msg = msg.decode()
            if msg[0] == wait_msg_encode[0] and msg[-1] == wait_msg_encode[-1]:
                msg = msg.decode()
            else:
                continue
            
            if msg == wait_msg and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender'.encode(), self.peer_addr)
                if self.verbose:
                    sys.stderr.write('[sender %s] Handshake success! '
                                     'Receiver\'s address is %s:%s\n' % (self.env_id, *addr))
                break

        self.sock.setblocking(0)  # non-blocking UDP socket
        
    def wavehand(self):
        """Wavehand with peer receiver. Must be called after run()."""
        if self.debug:
            sys.stderr.write('[sender %s] Wavehand Start\n' % (self.env_id))
            
        self.sock.setblocking(0)  # non-blocking UDP socket
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        
        # if self.env_id == 0:
        #     sys.stderr.write('[sender %s] Wavehand start!\n' % (self.env_id))

        self._retransmit('Bye from sender', 'Bye from receiver')
        
        # if self.env_id == 0:
        #     sys.stderr.write('[sender %s] Wavehand _retransmit quit!\n' % (self.env_id))
            
        self._waitquit('Bye ack from sender', 'Bye from receiver')
        if self.debug:
            sys.stderr.write('[sender %s] Wavehand OK\n' % (self.env_id))
        return
        
    def _waitquit(self, info, wait_info):
        MSL = 2000
        retry_times = 0
        wait_info_bytes = wait_info.encode()
        
        retransmit_flag = True
        while True:
            if retransmit_flag:
                self.sock.sendto(info.encode(), self.peer_addr)
            events = self.poller.poll(MSL)
            
            if not events:
                return
            else:
                retransmit_flag = False
                for fd, flag in events:
                    assert self.sock.fileno() == fd

                    if flag & ERR_FLAGS:
                        sys.exit('Channel closed or error occurred')

                    if flag & READ_FLAGS:
                        msg, addr = self.sock.recvfrom(1600)
                        # msg = msg.decode()
                        if addr == self.peer_addr:
                            if msg[0] == wait_info_bytes[0] and msg[-1] == wait_info_bytes[-1]:
                                if msg.decode() == wait_info:
                                    retransmit_flag = True
                                    retry_times += 1
                                    break 
                            else:
                                continue
                            
            if retry_times > 10:
                sys.stderr.write('[sender %s] Wavehand ack failed after 10 retries\n' % (self.env_id))
                raise Exception('[sender %s] wavehand ack fails' % (self.env_id))
            
                      
    def _retransmit(self, info, wait_info):
        retry_times = 0
        wait_info_bytes = wait_info.encode()
        
        retransmit_flag = True
        while True:
            if retransmit_flag:
                self.sock.sendto(info.encode(), self.peer_addr)
            events = self.poller.poll(TIMEOUT)

            if not events:  # not get ack and retransmit
                retransmit_flag = True
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write(
                        '[sender %s] Retransmission failed after 10 retries\n' % (self.env_id))
                    raise OSError('[sender %s] Retransmission fails' % (self.env_id))
                else:
                    if retry_times == 1:
                        sys.stderr.write('[sender %s] Wavehand timed out and retrying...\n' % (self.env_id))
                    continue
            else:
                retransmit_flag = False #not bye pks
                for fd, flag in events:
                    assert self.sock.fileno() == fd

                    if flag & ERR_FLAGS:
                        sys.exit('Channel closed or error occurred')

                    if flag & READ_FLAGS:
                        msg, addr = self.sock.recvfrom(1600)
                        # sys.stderr.write('[sender %s] handshake_others msg: %s\n' % (self.env_id, msg.decode()[:4]))
                        # msg = msg.decode()
                        if addr == self.peer_addr:
                            if msg[0] == wait_info_bytes[0] and msg[-1] == wait_info_bytes[-1]:
                                if msg.decode() == wait_info:#double compare
                                    # sys.stderr.write('[sender %s] Wavehand retransmit end\n' % (self.env_id))
                                    return 
                            else:
                                continue
                                
# ==============================================================================
#                      util function
# ============================================================================== 
                                
    def generate_expert_act_seq(self):
        '''
        Generate seq of expert or nn_act. True is expert, False is nn_act.
        Expert rate equaling 0 means that all act is nn_act.
        '''
        expert_rate = self.get_expert_rate()
        #+100 avoid index out of bound
        expert_act_seq = np.random.rand(Sender.max_steps+100) < expert_rate
        return expert_act_seq
        
    
    def get_expert_rate(self):
        expert_rate = 0
        return expert_rate