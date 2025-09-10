# ==============================================================================
#           Basic setting
# ==============================================================================  
_MODEL_PATH = './runs/A-trained_models/'
_DEBUG_LOG_PATH = './runs/debug_log'
_LOG_FILE_NAME = 'interval_'

# ==============================================================================
#           distribute setting
# ============================================================================== 
NODE0_LIST = []
NODE1_LIST = list(range(23))
NODE2_LIST = list(range(4))

NODES = {0: NODE0_LIST,
         1: NODE1_LIST,
         2: NODE2_LIST}
# ==============================================================================
#           Neural network setting
# ==============================================================================  
BATCH_SIZE = 4
EPOCH = 4 
LR = 2.5e-3
TRAIN_SETTING = (BATCH_SIZE, EPOCH, LR)

LSTM_HIDDEN_SIZE = 32
LSTM_LAYER = 1
CUDA_WORKER = False

LSTM_SIZE = (LSTM_LAYER, LSTM_HIDDEN_SIZE)
# ==============================================================================
#           RPC setting
# ==============================================================================  
ENV_NAME = "env_worker"
MODEL_NAME = "shared_model"
TRAINER_NAME = "trainer"

MODE_LIST = [TRAINER_NAME, MODEL_NAME, ENV_NAME]
FORMAT_SUFFIX = '_{}'
# ==============================================================================
#           DEBUG verbose
# ==============================================================================  
DEBUG_VERBOSE = True

# ==============================================================================
#           Configuration code dealing
# ==============================================================================
class Config:
    def __init__(self, load):
        # self._PATH_DEBUG_LOG = _DEBUG_LOG_PATH
        self._PATH_MODEL = _MODEL_PATH
        self._LOAD = load
        self._PATH_LOAD = None
        self._DEBUG_LOG_PATH = _DEBUG_LOG_PATH
        self._LOG_FILE_NAME = _LOG_FILE_NAME
        # self._DEVICE = _DEVICE
        # assert self._DEVICE in ['cpu', 'gpu']
    
        self._getTimeStamp()
        self._mkdir_path()
        self._get_load_path()
        
    def _getTimeStamp(self):
        from datetime import datetime
        self.strtime = datetime.now().strftime('D%m%d-t%H%M_%Ss_%Y')
        
        
    def _mkdir_path(self):
        import os
        self.run_path_model = os.path.join(self._PATH_MODEL, self.strtime)
        try:
            os.makedirs(self._DEBUG_LOG_PATH)
        except OSError:
            pass
        
    def _get_load_path(self):
        if self._LOAD:
            import os
            files = os.listdir(self._PATH_MODEL)
            #the 'D' is corresponded with 'D%m%d-t%H%M_%Ss_%Y'
            files = list(filter(lambda x: x[0] == 'D', files))
            files.sort()

            self._PATH_LOAD = os.path.join(self._PATH_MODEL, files[-1])
            
    def get_config_data(self):
        return self.run_path_model, self._PATH_LOAD, (self._DEBUG_LOG_PATH, self._LOG_FILE_NAME, DEBUG_VERBOSE)

# class RPCConfig:
#     def get_name(self):
#         return ENV_NAME, TRAINER_NAME
    
# ==============================================================================
#           Configuration Instance
# ==============================================================================
# rpc_config = RPCConfig()