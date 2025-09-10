import threading
import time, os, shutil, copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr.config import LSTM_SIZE
class Writer(SummaryWriter):
    def __init__(self, proc_env_dict, debug_log_info=None):
        super().__init__()
        self.proc_env_dict = proc_env_dict
        self.limit = int(len(proc_env_dict) * 2) 
        #when limit = proc*5,  150MB of disk 
         
        #env_id do not start at 0 such as [5,7,9] 
        #using index rpresents the pos in array, and index start at 0.
        #index corresponds the position in self.envs_write_num
        env_ids = {} 
        index = 0
        
        #one env may correspond 2 proc, so index don't mean proc
        #index is the relative pos about env_id
        for env_id in range(30):
            if env_id not in env_ids:
                env_ids[env_id] = index
                index += 1
        self.env_ids = env_ids
        
        self.main_steps = 0
        self.envs_write_num = [0] * len(self.env_ids) 
        self.envs_log_write_num = [0] * len(self.env_ids) 
        self.envs_log_err_num = [0] * len(self.env_ids) 
        
        self.reset_running_log()
        # self.log_path = None
        # if debug_log_info:
        #     self.log_path = os.path.join(*debug_log_info)
    
    def write_grade(self, res):
        # proc_relation is dynamic during distributed writing. Because the process of writing board does not set lock,
        # proc_relation may change if use a share pointer. Thus, we deepcopy the dict to avoid the problem.
        res, text, proc_relation = res
        for proc in range(len(self.proc_env_dict)):
            env_id = proc_relation[proc]
            indexofnum = self.env_ids[env_id]
            
            num = self.envs_write_num[indexofnum]
            
            tput, delay, loss, expert_acc, best_cwnd_rate, write_flag = res[:, proc]#i is proc
            if tput > 0 or delay > 0:
                #if results are logged., num = num + 1.
                self.add_scalar("Result-tput/env_id: %s" % (env_id), tput, num)
                self.add_scalar("Result-delay/env_id: %s" % (env_id), delay, num)
                self.add_scalar("Result-loss/env_id: %s" % (env_id), loss, num)
                self.add_scalar("Rate-acc/env_id: %s" % (env_id), expert_acc, num)
                self.add_scalar("Rate-best_cwnd/env_id: %s" % (env_id), best_cwnd_rate, num)
                self.envs_write_num[indexofnum] = num + 1 
            
            text_data = text[proc]
            if write_flag > 0 and text_data:
                self.write_text(proc, env_id, text_data)

    def write_text(self, proc_id, env_id, text_data):
        indexofnum = self.env_ids[env_id]
        step = self.envs_log_err_num[indexofnum]
            
        data = text_data
        self.add_text("err_log: env_id-%s" % (env_id), data, step)

        self.envs_log_err_num[indexofnum] = step + 1
                
#     def write_text(self, proc_id, env_id):
#         if self.log_path:
#             path_log = self.log_path + str(proc_id)
#             indexofnum = self.env_ids[env_id]
#             step = self.envs_log_err_num[indexofnum]
#             with open(path_log, "r+") as f:  
#                 data = f.readlines()  
#                 data = ' || '.join(data)
#                 self.add_text("err_log: env_id-%s" % (env_id), data, step)

#                 self.envs_log_err_num[indexofnum] = step + 1
#                 f.truncate(0) #clear file
    
    def write_main_log(self, text):
        self.add_text("A-main: " , text, self.main_steps)
        self.main_steps += 1
        
    
    def _write_running_log(self, running_log_tuple):
        running_log, running_log_dict, proc_relation = running_log_tuple
        proc_num, _, steps = running_log.shape
        # print('=====come log======')
        for proc in range(proc_num):
            if running_log_dict[proc] > 0:
                env_id = proc_relation[proc]
                indexofnum = self.env_ids[env_id]

                num = self.envs_log_write_num[indexofnum]
                for j in range(steps):
                    self.add_scalar("Run-rtts/env_id: %s" % (env_id), running_log[proc, 0, j], j)
                    self.add_scalar("Run-cwnd/env_id: %s" % (env_id), running_log[proc, 1, j], j)
                    # self.add_scalar("Run-delta_cwnd/env_id: %s" % (env_id), running_log[proc, 2, j], j)
                    
                    self.add_scalar("State-delay_rtt/env_id: %s" % (env_id), running_log[proc, 3, j], j)
                    self.add_scalar("State-delivery/env_id: %s" % (env_id), running_log[proc, 4, j], j)
                    self.add_scalar("State-send/env_id: %s" % (env_id), running_log[proc, 5, j], j)
                    self.add_scalar("State-tput-delta/env_id: %s" % (env_id), running_log[proc, 5, j] - running_log[proc, 4, j], j)
                    
                self.envs_log_write_num[indexofnum] = num + 1
    
    def write_running_log(self, running_log_tuple):
        self.log_count += 1
        if self.log_count > self.limit:
            self.reset_running_log()
        self._write_running_log(running_log_tuple)
    
    def reset_running_log(self):
        self.log_count = 0
        if self.file_writer:
            event_path = self.file_writer.event_writer._file_name
            try:
                os.remove(event_path)
            except Exception as e:
                print('delete failure: ', e)
            
            self.file_writer.close()
            self.file_writer = None

class Optimizer:
    def __init__(self, net, batch_size=4, epoch=4, lr=2.5e-3, \
                 device='cpu', writer=None):
        self._batch_size = batch_size
        self._epoch = epoch
        self._lr = lr
        self.device = torch.device(device if torch.cuda.is_available() \
                           and device != 'cpu' else "cpu")
        print('Optimizer device: ', self.device)
                
        self._net = net
        self._optimizer = torch.optim.Adam(self._net.parameters(),lr = self._lr)
        self._loss_function = nn.CrossEntropyLoss()
        self._writer = None #SummaryWriter()
        
        self._save_name = 'optimizer.pth'
        self._train_iter = 0
        self.set_writer(writer)
        
        if self.device.type != 'cpu':
            self._net.to(self.device)
    
    def set_writer(self, writer):
        self._writer = writer
    
    def train(self, data):
        X_train, y_train, m_train = data
        # if self.device.type != 'cpu':
        #     X_train, y_train, m_train = X_train.to(self.device), \
        #     y_train.to(self.device), m_train.to(self.device)
        
        loss_train = 0
        for epoch in range(self._epoch):
            minibatches = torch.randperm(len(X_train)).split(self._batch_size)
            # print('cuda train debug-epoch')
            for batch_idx, minibatch in enumerate(minibatches):
                x = X_train[minibatch]
                m = m_train[minibatch]
                
                y = y_train[minibatch].view(-1).to(self.device)
                out = self._net.train(x, m)
                
                loss = self._loss_function(out, y)
                # 清零梯度
                self._optimizer.zero_grad()
                # 计算梯度
                loss.backward()
                self._optimizer.step()
                loss_train += loss.item()
        # print('cuda train debug-leave')
        loss_avg_train = loss_train/(self._epoch * (batch_idx+1))
        
        if self._writer is not None:
            self._writer.add_scalar("A-Train/Loss", loss_avg_train, self._train_iter)
        self._train_iter += 1
        
    def save_opt(self, path):
        _save_name = os.path.join(path, self._save_name)
        torch.save(self._optimizer.state_dict(), _save_name)
        
    def load_opt(self, path):
        _save_name = os.path.join(path, self._save_name)
        self._optimizer.load_state_dict(torch.load(_save_name))
        
    def get_train_iteration(self):
        return self._train_iter
    
# In Buffer class, env_id means proc_id, env_id is a error name
class Buffer:
    '''
    Multiprocess sharing buffer
    '''
    def __init__(self, proc_num, state_dim, max_steps, action_dim, proc_relation):
        super().__init__()
        
        box_num = min(16, proc_num) # 5 is the test num
        self.max_step = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.box_num = box_num
        self.proc_num = proc_num
        self.proc_relation = proc_relation
        print('Box num: %s, proc num: %s' % (box_num, proc_num))
        
        self.test_flag = torch.Tensor([0])
        
        self.flags = torch.LongTensor([0]*box_num) #0 is idle; 1 is using; 2 is waiting train
        #dict: key-env_id, value-box_id 
        #env_id associates the using box_id
        self.env_box_ids = torch.LongTensor([-1]*proc_num)
                
        self.buffer_s = torch.zeros(box_num, self.max_step, self.state_dim) #state
        self.buffer_a = torch.zeros((box_num, self.max_step), dtype=torch.int64) #act
        self.buffer_m = torch.zeros(box_num, self.max_step, self.action_dim) #mask
        self.proc_train_num = torch.LongTensor([0] * proc_num)
        
        self.buffer_s_local = torch.zeros(self.proc_num, max_steps, self.state_dim) #state
        self.buffer_a_local = torch.zeros(self.proc_num, max_steps, dtype=torch.int64) #act
        self.buffer_m_local = torch.ones(self.proc_num, max_steps, self.action_dim) #mask
        
        #rtts cwnd
        self.running_log = torch.zeros(proc_num, 6, self.max_step) 
        self.running_log_dict = torch.zeros(proc_num)
        
        self.result = -1 * torch.ones((6, proc_num))
        self.txt_result = [''] * proc_num

        
#     def share_memory(self):
#         self.buffer_s.share_memory_()
#         self.buffer_a.share_memory_()
#         self.buffer_m.share_memory_()
        
#         self.result.share_memory_()
#         self.proc_train_num.share_memory_()
        
#         self.flags.share_memory_()
#         self.env_box_ids.share_memory_()
#         self.test_flag.share_memory_()
    
    # def create_local_tensor(self):
    #     self.buffer_s_local = torch.zeros(self.proc_num, self.max_step, self.state_dim) #state
    #     self.buffer_a_local = torch.zeros((self.proc_num, self.max_step), dtype=torch.int64) #act
    #     self.buffer_m_local = torch.zeros(self.proc_num, self.max_step, self.action_dim) #mask
    #     self.lstm_h0_c0 = [None] * self.proc_num
 
        
#     def get_lstm_h0(self, step_cnt, proc_id, lstm_size):
#         if step_cnt > 0:
#             lstm_hidden_local = self.lstm_h0_c0[proc_id]
#         else:
#             # lstm_size = (LSTM_LAYER, LSTM_HIDDEN_SIZE)
#             lstm_hidden_local = (torch.zeros(lstm_size), torch.zeros(lstm_size))
#         return lstm_hidden_local
    
#     def set_lstm_h0(self, proc_id, lstm_h0):
#         self.lstm_h0_c0[proc_id] = lstm_h0      
                        
    # def get_act(self, state_act_pair):
    #     # state, expert, mask, step_cnt, env_id = state_act_pair
    #     expert = state_act_pair[1]
    #     self.save_buffer(state_act_pair)
    #     return expert

#     def save_buffer(self, state_act_pair):
#         step_cnt, proc_id = state_act_pair[-2:]
#         self._save_local(state_act_pair)
        
#         if step_cnt == self.max_step-1:
#             box_id = self._apply_resource(proc_id)
#             # exist idle box
#             if box_id is not None:
#                 self._local_to_buffer(box_id, proc_id)
#                 self._close_resource(proc_id, box_id)
                
#                 self.proc_train_num[proc_id] += 1
    
    @staticmethod
    def save_buffer_rpc(buf, proc_id, states, acts, masks):
        self = buf.local_value()

        box_id = self._apply_resource(proc_id)
        # exist idle box
        if box_id is not None:
            self._local_to_buffer(box_id, proc_id, states, acts, masks)
            self._close_resource(proc_id, box_id)

            self.proc_train_num[proc_id] += 1

    @staticmethod
    def save_buffer_rpc_segment(buf, proc_id, step_info, states, acts, masks):
        self = buf.local_value()
        
        step, seqlen = step_info
        end = step + seqlen
        self.buffer_s_local[proc_id, step:end] = states
        self.buffer_a_local[proc_id, step:end] = acts
        self.buffer_m_local[proc_id, step:end] = masks
        
        if end == self.max_step:
            box_id = self._apply_resource(proc_id)
            # exist idle box
            if box_id is not None:
                self._local_to_buffer(box_id, proc_id)
                self._close_resource(proc_id, box_id)

                self.proc_train_num[proc_id] += 1

                
    # def _save_local(self, state_act_pair):
    #     state, expert, mask, step_cnt, proc_id = state_act_pair
    #     self.buffer_s_local[proc_id, step_cnt] = torch.Tensor(state)
    #     self.buffer_a_local[proc_id, step_cnt] = expert
    #     self.buffer_m_local[proc_id, step_cnt] = torch.Tensor(mask)
    
    def _local_to_buffer(self, box_id, proc_id):
        self.buffer_s[box_id] = self.buffer_s_local[proc_id, :]
        self.buffer_a[box_id] = self.buffer_a_local[proc_id, :]
        self.buffer_m[box_id] = self.buffer_m_local[proc_id, :]
        
    
    def is_buffer_full(self):
        is_full = False
        if self.flags.sum() == self.box_num * 2:
            is_full = True
        self.is_full = is_full
        return is_full
    
    def get_samples(self):
        assert self.is_full is True
        samples = (self.buffer_s.clone(), \
                   self.buffer_a.clone(), \
                   self.buffer_m.clone())
        self._clear_box()
        self.is_full = False
        return samples
            
    def _apply_resource(self, env_id):
        pos = self.flags.argmin() 
        if self.flags[pos] == 0:
            self._set_box(env_id, pos)
        else:
            pos = None
        return pos
               
    def _close_resource(self, env_id, box_id):
        self.flags[box_id] = 2 #waiting train
        self.env_box_ids[env_id] = -1 
    
    def _set_box(self, env_id, pos):
        self.flags[pos] = 1 #loading 
        self.env_box_ids[env_id] = pos
        
    def _clear_box(self):
        self.flags[:] = 0
        # self.env_box_ids[:] = -1 
        #can't clear all env_ids if the box num does not equal proc num
        #because the env_box_id is used to determine whether the proc is loading.in func 
        #_get_resource_id

    @staticmethod
    def log_result_all_rpc(buf_rref, proc_id, env_id, res):
        self = buf_rref.local_value()
        result, log, text = res
        self.log_result(proc_id, result)
        self.log_running(proc_id, log)
        self.log_txt(proc_id, text)
        self.proc_relation[proc_id] = env_id
    
    def log_result(self, proc_id, res):
        self.result[:, proc_id] = torch.Tensor(res)
        
    def log_txt(self, proc_id, text):
        self.txt_result[proc_id] = text
        
        
    def log_running(self, proc_id, log):
        self.running_log_dict[proc_id] = 1
        self.running_log[proc_id] = torch.Tensor(log)

    def get_running_log_and_clear(self):
        running_log = self.running_log.clone()
        running_log_dict = self.running_log_dict.clone()
        proc_relation = copy.deepcopy(self.proc_relation)

        self.running_log_dict[:] = 0
        self.running_log[:] = 0
        
        return (running_log, running_log_dict, proc_relation)
    
    def get_result_and_clear(self):
        res = self.result.clone()
        text = copy.deepcopy(self.txt_result)
        proc_relation = copy.deepcopy(self.proc_relation)
        
        self.result[:] = -1
        self.txt_result = [''] * self.proc_num
        return res, text, proc_relation

    
class LocalBuffer:
    '''
    A buffer for RPCWokerLocal class.
    '''
    def __init__(self, obs_dim, action_dim, max_steps):
        self.buffer_s_local = torch.zeros(max_steps, obs_dim) #state
        self.buffer_a_local = torch.zeros(max_steps, dtype=torch.int64) #act
        self.buffer_m_local = torch.ones(max_steps, action_dim) #mask
        self.lstm_h0_c0 = None
    
    def save_state_tuple(self, state, expert, mask, step_cnt):
        self.buffer_s_local[step_cnt] = torch.Tensor(state)
        self.buffer_a_local[step_cnt] = expert
        self.buffer_m_local[step_cnt] = torch.Tensor(mask)
        
    def get_tuples(self):
        return self.buffer_s_local, self.buffer_a_local, self.buffer_m_local
    
    def get_tuples_index(self, start, end):
        states = self.buffer_s_local[start:end]
        acts = self.buffer_a_local[start:end]
        masks = self.buffer_m_local[start:end]
        return states, acts, masks  
        
    
    def get_lstm_h0(self, step_cnt):
        if step_cnt > 0:
            lstm_hidden_local = self.lstm_h0_c0
        else:
            lstm_hidden_local = init_lstm_h0()
        return lstm_hidden_local
    
    def set_lstm_h0(self, lstm_h0):
        self.lstm_h0_c0 = lstm_h0 

#===========================================================
#           Helper function and class
#===========================================================
class WriterThread(threading.Thread):
    '''
    A thread for writing tensorboard.
    '''
    def __init__(self, trainer):
        super().__init__(daemon=True)
        
        self.trainer = trainer
        
    def run(self):
        while 1:
            time.sleep(2) 
            self.trainer.write_board()

def init_lstm_h0():
    hx = (torch.zeros(LSTM_SIZE), torch.zeros(LSTM_SIZE))
    return hx
