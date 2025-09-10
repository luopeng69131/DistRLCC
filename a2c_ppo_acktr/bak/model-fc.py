#MODEL FC
import time, os, shutil 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# PATH = './trained_models/'
# DEVICE = 'cpu'

class Trainer:
    def __init__(self, envs, device, model_path, debug_log_info):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() \
                                   and device == 'gpu' else "cpu")
        self.path = model_path
        
        self._num_envs = envs.num_envs
        self._state_dim = envs.observation_space
        self._action_dim = envs.action_space
        self._max_steps = envs.value_observation_space
        
        self.buffer = Buffer(self._num_envs, self._state_dim, self._max_steps, self._action_dim)
        self.worker_net = Agent(self._state_dim, self._action_dim, 'test')
        self.train_net = Agent(self._state_dim, self._action_dim, 'train')

        self.writer = Writer(envs.proc_env_relation, debug_log_info)
        time.sleep(1)
        self.running_logger = Writer(envs.proc_env_relation)
        self.opt = Optimizer(net=self.train_net, device=device, writer=self.writer)
        
    def train(self):
        print('start train!')
        while 1:
            time.sleep(2)
            self.write_board()
            
            if self.buffer.is_buffer_full():
                print('Full!----------')
                buf = self.buffer.get_samples()

                self._train_model(buf)
                self.update_worker()
                
                print('Train num: ', self.buffer.proc_train_num)
                
                if (self.opt.get_train_iteration()+1) % 25 == 0:
                    self.save()
                    
                    
    
    def update_worker(self):
        self.worker_net.load_state_dict(self.train_net.state_dict())
        
    def write_board(self):
        res = self.buffer.get_result_and_clear()
        running_log = self.buffer.get_running_log_and_clear()
        
        self.writer.write_grade(res)
        
        _, running_log_dict = running_log
        if running_log_dict.sum() > 0:
            self.running_logger.write_running_log(running_log)
            
        # if running_log is not None:
        #     self.writer.write_running_log(running_log)
        
    def _train_model(self, data):
        print('train epoch')
        self.opt.train(data)
        print('-------train end-------')
    
    def get_share_worker(self):
        worker = ShareWorker(self.buffer, self.worker_net)
        worker.share_memory()
        return worker
    
    def save(self):
        try:
            os.makedirs(self.path)
        except OSError:
            pass
        self.opt.save_opt(self.path)
        self.train_net.save_model(self.path)
        print('-----save model!-----')

    def load(self, load_path):
        self.opt.load_opt(load_path)
        self.train_net.load_model(load_path)
        self.update_worker()
        print("Loading model success from path '%s' !" % load_path)
        
    def close(self):
        self.writer.close()
        self.running_logger.close()
        self.save()
        
        
class Optimizer:
    def __init__(self, net, batch_size=8192, epoch=4, lr=2.5e-3, \
                 device='cpu', writer=None):
        self._batch_size = batch_size
        self._epoch = epoch
        self._lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() \
                           and device == 'gpu' else "cpu")
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
        if self.device.type != 'cpu':
            X_train, y_train, m_train = X_train.to(self.device), \
            y_train.to(self.device), m_train.to(self.device)
        
        loss_train = 0
        for epoch in range(self._epoch):
            # print("\r","Epoch =======>",epoch, end='', flush= True)
            # loss_epoch = 0
            minibatches = torch.randperm(len(X_train)).split(self._batch_size)
            for batch_idx, minibatch in enumerate(minibatches):
                x = X_train[minibatch]
                y = y_train[minibatch]
                m = m_train[minibatch]

                out = self._net(x, m)
                loss = self._loss_function(out, y)
                # 清零梯度
                self._optimizer.zero_grad()
                # 计算梯度
                loss.backward()
                self._optimizer.step()
                loss_train += loss.item()
            # loss_epoch_avg = loss_epoch/(batch_idx+1)
        loss_avg_train = loss_train/(self._epoch * (batch_idx+1))
        # self._train_loss.append(loss_epoch_avg)
        
        if self._writer is not None:
            self._writer.add_scalar("Loss/train", loss_avg_train, self._train_iter)
        self._train_iter += 1
        
    def save_opt(self, path):
        _save_name = os.path.join(path, self._save_name)
        torch.save(self._optimizer.state_dict(), _save_name)
        
    def load_opt(self, path):
        _save_name = os.path.join(path, self._save_name)
        self._optimizer.load_state_dict(torch.load(_save_name))
        
    def get_train_iteration(self):
        return self._train_iter
        
    
class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, mode=None):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self._save_name = 'model.pth'
        
        # self.bn = nn.BatchNorm1d(obs_dim)
        self.fc1 = nn.Linear(obs_dim, 64)
        self.output = nn.Linear(64, action_dim)
        
        assert mode in ['train', 'test']
        if mode == 'train':
            self.train()
        else:
            self.eval()

    def forward(self, inputs, mask):
        x = inputs
        x = F.relu(self.fc1(x))
        x = self.output(x) + mask
        return x
    
    def get_action(self, state, mask):
        with torch.no_grad():
            # self.eval()
            input_tensor, mask_tensor = self.view_tensor(state, mask)
            out = self.forward(input_tensor, mask_tensor)
            act = out[0].argmax()
        return act.item()
    
    def view_tensor(self, inputs, mask):
        inputs_tensor = torch.tensor([inputs], requires_grad=False)
        mask_tensor = torch.tensor([mask], requires_grad=False)
        return inputs_tensor, mask_tensor
    
    def save_model(self, path):
        _save_name = os.path.join(path, self._save_name)
        torch.save(self.state_dict(), _save_name)
        
    def load_model(self, path):
        _save_name = os.path.join(path, self._save_name)
        self.load_state_dict(torch.load(_save_name))
        
#========================running module==================================
class ShareWorker:
    '''
    Interface shared worker for envs: buf and worker model 
    '''
    def __init__(self, buf, model):
        self.buf = buf
        self.model = model
    
    def get_act(self, state_act_pair):
        state, expert, mask, step_cnt, env_id = state_act_pair
        act = self.model.get_action(state, mask)
        expert = self.buf.get_act(state_act_pair)
        return act
    
    def share_memory(self):
        self.buf.share_memory()
        # self.model.share_memory()
        
    def set_lock(self, lock):
        self.buf.set_lock(lock)
    
    def log_result(self, proc_id, res):
        result, log =  res
        self.buf.log_result(proc_id, result)
        self.buf.log_running(proc_id, log)
        
    def cpl(self):
        self.buf.cpl()    
    
# In Buffer class, env_id means proc_id, env_id is a error name
class Buffer(nn.Module):
    '''
    Multiprocess sharing buffer
    '''
    def __init__(self, proc_num, state_dim, max_steps, action_dim):
        super().__init__()
        
        box_num = min(16, proc_num) # 5 is the test num
        self.max_step = max_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.box_num = box_num
        self.proc_num = proc_num
        print('Box num: %s, proc num: %s' % (box_num, proc_num))
        
        # self._lock = mp.Lock()
        self.test_flag = torch.Tensor([0])
        
        self.flags = torch.LongTensor([0]*box_num) #0 is idle; 1 is using; 2 is waiting train
        #dict: key-env_id, value-box_id 
        #env_id associates the using box_id
        self.env_box_ids = torch.LongTensor([-1]*proc_num)
        
        self.buffer_s_local = torch.zeros(self.max_step, self.state_dim) #state
        self.buffer_a_local = torch.zeros((self.max_step), dtype=torch.int64) #act
        self.buffer_m_local = torch.zeros(self.max_step, self.action_dim) #mask
        self.local_train_num = 0      
        
        self.buffer_s = torch.zeros(box_num, self.max_step, self.state_dim) #state
        self.buffer_a = torch.zeros((box_num, self.max_step), dtype=torch.int64) #act
        self.buffer_m = torch.zeros(box_num, self.max_step, self.action_dim) #mask
        self.proc_train_num = torch.LongTensor([0] * proc_num)
        
        #rtts cwnd
        self.running_log = torch.zeros(proc_num, 3, self.max_step) 
        self.running_log_dict = torch.zeros(proc_num)
        # self.running_cwnd = torch.zeros(box_num, self.max_step)
        
        self.result = -1 * torch.ones((4, proc_num))
        
    def share_memory(self):
        self.buffer_s.share_memory_()
        self.buffer_a.share_memory_()
        self.buffer_m.share_memory_()
        
        self.result.share_memory_()
        self.proc_train_num.share_memory_()
        
        self.flags.share_memory_()
        self.env_box_ids.share_memory_()
        self.test_flag.share_memory_()
    
    def set_lock(self, lock):
        self._lock = lock
        
    def get_act(self, state_act_pair):
        # state, expert, mask, step_cnt, env_id = state_act_pair
        expert = state_act_pair[1]
        self.save_buffer(state_act_pair)
        return expert

    def save_buffer(self, state_act_pair):
        step_cnt, env_id = state_act_pair[-2:]
        self._save_local(state_act_pair)
        
        if step_cnt == self.max_step-1:
            box_id = self._apply_resource(env_id)
            # exist idle box
            if box_id is not None:
                self._local_to_buffer(box_id)
                self._close_resource(env_id, box_id)
                
                self.local_train_num += 1
                self.proc_train_num[env_id] = self.local_train_num
            # print('proc_id: %s | train num: %s' % (env_id, self.local_train_num))
                
    def _save_local(self, state_act_pair):
        state, expert, mask, step_cnt, env_id = state_act_pair
        self.buffer_s_local[step_cnt] = torch.Tensor(state)
        self.buffer_a_local[step_cnt] = expert
        self.buffer_m_local[step_cnt] = torch.Tensor(mask)
    
    def _local_to_buffer(self, box_id):
        self.buffer_s[box_id] = self.buffer_s_local[:]
        self.buffer_a[box_id] = self.buffer_a_local[:]
        self.buffer_m[box_id] = self.buffer_m_local[:]
        
    
    def is_buffer_full(self):
        is_full = False
        if self.flags.sum() == self.box_num * 2:
            is_full = True
        self.is_full = is_full
        return is_full
    
    def get_samples(self):
        assert self.is_full is True
        samples = (self.buffer_s.clone().view(-1,self.state_dim), \
                   self.buffer_a.clone().view(-1), \
                   self.buffer_m.clone().view(-1,self.action_dim))
        with self._lock:
            self._clear_box()
            self.is_full = False
        return samples
            
    def _apply_resource(self, env_id):
        with self._lock:
            # print(env_id,  self.flags)
            pos = self.flags.argmin() 
            # find idle box, and idle box flag is 0.
            # print('env: ', env_id, '| pos: ', pos)
            if self.flags[pos] == 0:
                self._set_box(env_id, pos)
            else:
                pos = None
        return pos
    
    def _get_resource_id(self, env_id):
        pos = self.env_box_ids[env_id]
        if pos == -1:
            pos = None
        return pos
    
    
    def _save_tuple(self, box_id, state, act, mask, step_cnt):
        self.buffer_s[box_id, step_cnt] = torch.Tensor(state)
        self.buffer_a[box_id, step_cnt] = act
        self.buffer_m[box_id, step_cnt] = torch.Tensor(mask)
        
    def _close_resource(self, env_id, box_id):
        with self._lock:
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
    
        
    def cpl(self):
        print('cpl arrive flag')
        with self._lock:
            print('get lock')
            time.sleep(10)
            self.test_flag.add_(4) #in place op
            print('release lock')
        
    
    def log_result(self, proc_id, res):
        self.result[:, proc_id] = torch.Tensor(res)
        
    def log_running(self, proc_id, log):
        self.running_log_dict[proc_id] = 1
        self.running_log[proc_id] = torch.Tensor(log)

    def get_running_log_and_clear(self):
        running_log = self.running_log.clone()
        running_log_dict = self.running_log_dict.clone()

        self.running_log_dict[:] = 0
        self.running_log[:] = 0
        return (running_log, running_log_dict)
    
        
    def get_result_and_clear(self):
        res = self.result.clone()
        self.result[:] = -1
        return res

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
        for proc_id, env_id in proc_env_dict.items():
            if env_id not in env_ids:
                env_ids[env_id] = index
                index += 1
        self.env_ids = env_ids
        
        self.envs_write_num = [0] * len(self.env_ids) 
        self.envs_log_write_num = [0] * len(self.env_ids) 
        self.envs_log_err_num = [0] * len(self.env_ids) 
        
        self.reset_running_log()
        self.log_path = None
        if debug_log_info:
            self.log_path = os.path.join(*debug_log_info)

        
    
    def write_grade(self, res):
        for proc in range(len(self.proc_env_dict)):
            env_id = self.proc_env_dict[proc]
            indexofnum = self.env_ids[env_id]
            
            num = self.envs_write_num[indexofnum]
            
            tput, delay, loss, write_flag = res[:, proc]#i is proc
            if tput > 0 or delay > 0:
                #if results are logged., num = num + 1.
                self.add_scalar("Result-tput/env_id: %s" % (env_id), tput, num)
                self.add_scalar("Result-delay/env_id: %s" % (env_id), delay, num)
                self.add_scalar("Result-loss/env_id: %s" % (env_id), loss, num)
                self.envs_write_num[indexofnum] = num + 1 
            
            if write_flag > 0:
                self.write_text(proc, env_id)
    
    def write_text(self, proc_id, env_id):
        if self.log_path:
            path_log = self.log_path + str(proc_id)
            indexofnum = self.env_ids[env_id]
            step = self.envs_log_err_num[indexofnum]
            with open(path_log, "r+") as f:  
                data = f.readlines()  
                data = ' ; '.join(data)
                self.add_text("err_log: env_id-%s" % (env_id), data, step)

                self.envs_log_err_num[indexofnum] = step + 1
                f.truncate(0) #clear file
    
    def _write_running_log(self, running_log_tuple):
        running_log, running_log_dict = running_log_tuple
        proc_num, _, steps = running_log.shape
        # print('=====come log======')
        for proc in range(proc_num):
            if running_log_dict[proc] > 0:
                env_id = self.proc_env_dict[proc]
                indexofnum = self.env_ids[env_id]

                num = self.envs_log_write_num[indexofnum]
                for j in range(steps):
                    self.add_scalar("RunLog-rtts/env_id: %s" % (env_id), running_log[proc, 0, j], j)
                    self.add_scalar("RunLog-cwnd/env_id: %s" % (env_id), running_log[proc, 1, j], j)
                    self.add_scalar("RunLog-delta_cwnd/env_id: %s" % (env_id), running_log[proc, 2, j], j)
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
            
        
        
            
        
        
    

        
        
        
                

 