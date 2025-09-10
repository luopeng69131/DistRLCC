#MODEL FC CHANGED TO LSTM
import time, os, shutil, copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from a2c_ppo_acktr.utils import get_now_time_str, init_rnn_orthogonal
from a2c_ppo_acktr.modeltool import Writer, Optimizer, Buffer, WriterThread, init_lstm_h0
from a2c_ppo_acktr.config import TRAIN_SETTING, LSTM_HIDDEN_SIZE, LSTM_LAYER, LSTM_SIZE, CUDA_WORKER


class Trainer:
    def __init__(self, envs, worker_models, device, model_path, debug_log_info, load_path=None):
        
        self.device = torch.device(device if torch.cuda.is_available() \
                                   and device != 'cpu' else "cpu")
        self.path = model_path
        self.load_path = load_path
        
        self._num_envs = envs.num_envs
        self._state_dim = envs.observation_space
        self._action_dim = envs.action_space
        self._max_steps = envs.value_observation_space
        
        self.proc_env_relation = envs.proc_env_relation
        
        self.buffer = Buffer(self._num_envs, self._state_dim, self._max_steps, self._action_dim, self.proc_env_relation)
        
        # worker_dev = device if CUDA_WORKER else 'cpu'
        # self.worker_net = Agent(self._state_dim, self._action_dim, 'test', worker_dev)
        self.train_net = Agent(self._state_dim, self._action_dim, 'train', device)
        self.worker_models = worker_models
        # print('Worker device: %s' % self.worker_net.device)

        self.writer = Writer(envs.proc_env_relation, debug_log_info)
        time.sleep(1)
        self.running_logger = Writer(envs.proc_env_relation)
        self.opt = Optimizer(self.train_net, *TRAIN_SETTING, device, self.writer)
        self.write_board_thread = WriterThread(self)
        if self.load_path:
            self.load(load_path)
        
        self.write_basic_info()
        
    def train(self):
        print('start train!')
        self.write_board_thread.start()
        while 1:
            time.sleep(0.1)
            if self.buffer.is_buffer_full():
                print('Full!----------')
                buf = self.buffer.get_samples()

                self._train_model(buf)
                self.update_worker()
                
                print('Train num: ', self.buffer.proc_train_num)
                
                if (self.opt.get_train_iteration()+1) % 25 == 0:
                    self.save()  

    
    def update_worker(self):
        # self.worker_net.load_state_dict(self.train_net.state_dict())
        self.worker_models.update_weight(self.train_net.net.state_dict())
        
    def write_board(self):
        res = self.buffer.get_result_and_clear()
        running_log = self.buffer.get_running_log_and_clear()
        self.writer.write_grade(res)
        
        running_log_dict = running_log[1]
        if running_log_dict.sum() > 0:
            self.running_logger.write_running_log(running_log)
            
    def write_basic_info(self):
        # strtime = datetime.now().strftime('%Y/%m/%d-%H:%M:%Ss')
        text = 'Start time: ' + get_now_time_str() 
        self.writer.write_main_log(text)
        text = 'Relation of proc and env: ' + self.writer.proc_env_dict.__str__()
        self.writer.write_main_log(text)
        text = 'batch: %s, epoch: %s, lr: %.3e, train device: %s' % (self.opt._batch_size, self.opt._epoch, self.opt._lr, self.opt.device.__str__())
        self.writer.write_main_log(text)
        text = 'proc_num: %s, box_num: %s' % (self.buffer.proc_num, self.buffer.box_num)
        self.writer.write_main_log(text)
        text = 'Save model path: %s; loading model: %s.' %(self.path, self.load_path)
        self.writer.write_main_log(text)
        text = 'Net model: rnn_layer-% s, rnn_hidden- %s || %s .'  % (*LSTM_SIZE, self.train_net.net.__str__())
        self.writer.write_main_log(text)
        
        # text
        # dummy = (torch.ones(1,self._state_dim), torch.ones((1,self._action_dim)), torch.ones(_LSTM_SIZE), torch.ones(_LSTM_SIZE))
        # # if set torch.jit.script about net, it will print warning: already script module 
        # self.writer.add_graph(self.worker_net.net, dummy)
        
    def _train_model(self, data):
        print('train epoch')
        self.opt.train(data)
        print('-------train end-------')
    
    def get_share_buf(self):
        # worker = ShareWorker(self.worker_net)
        return self.buffer
    
    def save(self):
        try:
            os.makedirs(self.path)
        except OSError:
            pass
        self.opt.save_opt(self.path)
        self.train_net.save_model(self.path)
        print('-----save model!-----')

    def load(self, load_path):
        self.load_path = load_path
        self.opt.load_opt(load_path)
        self.train_net.load_model(load_path)
        self.update_worker()
        print("Loading model success from path '%s' !" % load_path)
        
    def close(self):
        self.save()
        self.writer.write_main_log('Env train num: ' + self.buffer.proc_train_num.tolist().__str__())
        self.writer.write_main_log('End time:' + get_now_time_str())
        self.writer.close()
        self.running_logger.close()

        
class ShareWorker:
    '''
    Interface shared worker for envs: buf and worker model 
    '''
    def __init__(self, model, env_info=None):
        if model is None:
            assert env_info is not None
            device = 'gpu' if CUDA_WORKER else 'cpu'
            worker_dev = torch.device("cuda:0" if torch.cuda.is_available() \
                                   and device == 'gpu' else "cpu")
            obs_dim, act_dim = env_info[:-1]
            self.model = Agent(obs_dim, act_dim, 'test', worker_dev)
            # print('shared_model dev: ' + worker_dev.type)
        else:
            assert model is not None
            self.model = model
        # self.init_local_var()
            
    
    @staticmethod
    def get_act_rpc(model_rref, state_act_pair):
        self = model_rref.local_value()
        result = self.get_act()
        return result
        
    def get_act(self, state_act_pair):
        (state, lstm_h0), expert, mask, step_cnt, proc_id = state_act_pair
        
        act, next_lstm_h0_c0 = self.model.get_action(state, mask, lstm_h0)
        return (act, expert, next_lstm_h0_c0)
        
    
    @staticmethod
    def update_model_rpc(model_rref, model_state_dict):
        self = model_rref.local_value()
        # print('update weight!')
        self.model.net.load_state_dict(model_state_dict)
    
    # def init_local_var(self):
    #     self.buf.create_local_tensor()
    

    # @staticmethod
    # def log_result_rpc(model_rref, proc_id, res):
    #     self = model_rref.local_value()
    #     result, log, text = res
    #     self.buf.log_result(proc_id, result)
    #     self.buf.log_running(proc_id, log)
    #     self.buf.log_txt(proc_id, text)


class TestModel:
    '''
    The class implements that a model tests in testset. Its function is like to class RPCWokerLocal,
    but RPCWokerLocal is used in train.
    '''
    def __init__(self, env_info, load_path=None):
        self.model = ShareWorker(None, env_info)
        if load_path is not None:
            self.load(load_path, 'cpu')
    
    def get_act(self, state_act_pair):
        state, expert, mask, step_cnt, proc_id = state_act_pair
        
        h0_c0 = self._get_lstm_h0(step_cnt)
        
        state_act_pair = ((state, h0_c0), expert, mask, step_cnt, proc_id)
        act, expert, h0_c0 = self.model.get_act(state_act_pair)
        
        self._set_lstm_h0(h0_c0)
        return (act, expert)
        
    def _get_lstm_h0(self, step_cnt):
        if step_cnt > 0:
            lstm_hidden_local = self.lstm_h0_c0
        else:
            lstm_hidden_local = init_lstm_h0()
        return lstm_hidden_local 
    
    def _set_lstm_h0(self, lstm_h0):
        self.lstm_h0_c0 = lstm_h0
    
    def load(self, load_path, device_str):
        self.load_path = load_path
        
        device = torch.device(device_str)
        self.model.model.load_model(load_path, device)
        print('load test model succeeds!')
        
    
#========================running module==================================

  
class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, mode=None, device='cpu'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() \
                           and device != 'cpu' else "cpu")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self._save_name = 'model.pth'
        # self.net = torch.jit.script(RNNNet(obs_dim, action_dim)).to(self.device)
        self.net = RNNNet(obs_dim, action_dim).to(self.device)
        
        assert mode in ['train', 'test']
        if mode == 'train':
            self.net.train()
        else:
            self.net.eval()
            
    def forward(self, inputs, mask, lstm_h0_c0):
        norm_input = self._normalize(inputs)
        x, hx = self.net(norm_input, mask, *lstm_h0_c0)
        return x, hx
    
    def _normalize(self, state):
        # state: basic (4) + new_add (1) + one_hot(5)
        norm = state / torch.Tensor([200.0, 200.0, 200.0, 5000.0, 100.0, \
                                     1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
        # print('1', end='')
        return norm
    
    def train(self, inputs, mask):
        # inputs: N * L * H, N is the batch, L is the seq and H is the feature size
        h0 = torch.zeros(LSTM_LAYER, inputs.shape[0], LSTM_HIDDEN_SIZE).to(self.device)
        c0 = torch.zeros(LSTM_LAYER, inputs.shape[0], LSTM_HIDDEN_SIZE).to(self.device)
        inputs = inputs.to(self.device)
        mask = mask.view(-1, self.action_dim).to(self.device)
        
        x, _ = self.forward(inputs, mask, (h0, c0))
        return x
    
    def get_action(self, state, mask, lstm_h0_c0):
        #state: (H,) 
        with torch.no_grad():
            input_tensor, mask_tensor = self.view_tensor(state, mask)
            h0, c0 = lstm_h0_c0
            
            input_tensor = input_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device) 
            h0, c0 = h0.to(self.device), c0.to(self.device)
            
            out, (h0_next, c0_next) = self.forward(input_tensor, mask_tensor, (h0, c0))
            act = out[0].argmax().item()
        return act, (h0_next.to('cpu'), c0_next.to('cpu'))
    
    def view_tensor(self, inputs, mask):
        # reshape state (H,) -> (1, H), this shape can be taken by LSTM
        inputs_tensor = torch.tensor([inputs], requires_grad=False)
        mask_tensor = torch.tensor([mask], requires_grad=False)
        return inputs_tensor, mask_tensor
    
    def save_model(self, path):
        _save_name = os.path.join(path, self._save_name)
        torch.save(self.net.state_dict(), _save_name)
        
    def load_model(self, path, device=None):
        _save_name = os.path.join(path, self._save_name)
        if device is None:
            self.net.load_state_dict(torch.load(_save_name))
        else:
            self.net.load_state_dict(torch.load(_save_name, map_location=device))

        
class RNNNet(nn.Module):
    lstm_hidden: int
    num_layers: int
    def __init__(self, obs_dim, action_dim):
        super().__init__()   
        self.lstm_hidden = LSTM_HIDDEN_SIZE
        self.num_layers = LSTM_LAYER
        self.rnn = nn.LSTM(obs_dim, self.lstm_hidden, num_layers=self.num_layers, batch_first=True)
        self.output = nn.Linear(self.lstm_hidden, action_dim)
        
        init_rnn_orthogonal(self.rnn)
        
        
    def forward(self, inputs, mask, h0, c0):
        x, hx = self.rnn(inputs, (h0, c0))
        x = x.contiguous().view(-1, self.lstm_hidden)
        x = self.output(x) + mask
        return x, hx

