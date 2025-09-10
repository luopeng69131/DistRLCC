# more machine program, one is trainer, another is envs and model;
# However the delay err ratio in steps is about 10-20%.
# It cannot be satisfied by the requirements.
# Adjusting rpc thread and shared_worker number cannot work.
# LocalTest is ok, delay err ratio in steps is about 1%.

#master: python main_mach_test_load.py --gpu 0 --node-index 0  --node-num 2 --IP 192.168.0.104
#slave: python main_mach_test_load.py --gpu 0 --node-index 1  --node-num 2 --total-worker 6 --IP 192.168.0.104 
# the only distinction about master and slave is the node index
import time
import argparse

import torch.multiprocessing as mp

# from a2c_ppo_acktr.game.envs import make_vec_envs
from a2c_ppo_acktr import rpc_rl
from a2c_ppo_acktr.model import Trainer
from a2c_ppo_acktr.utils import pkill, assign_env_id, set_ip_forwarding
from a2c_ppo_acktr.config import Config, ENV_NAME, MODEL_NAME, TRAINER_NAME, MODE_LIST, NODES

# !!! After reboot, must run 'sudo sysctl -w net.ipv4.ip_forward=1' firstly
# tensorboard --logdir './runs' --host 0.0.0.0 --port 6006
# http://192.168.0.103:6006
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--load',
        default=False,
        action='store_true',
        help='load a trained model')
    parser.add_argument(
        '--tun',
        default=False,
        action='store_true',
        help='load a trained model')
    parser.add_argument(
        '--gpu',
        type=int, 
        default=-1,
        help='-1 represents using cpu, >=0 is the index of cuda ')
    # parser.add_argument(
    #     '--env-num', 
    #     type=int, 
    #     default=2,
    #     help='world size for RPC, rank 0 is the agent, others are observers')
    parser.add_argument(
        '--node-num', 
        type=int, 
        default=2,
        help='Env number of every node , rank 0 is the agent, others are observers')
    parser.add_argument(
        '--node-index', 
        type=int, 
        help='env node index')
    # parser.add_argument(
    #     '--local-proc', 
    #     type=int, 
    #     default=-1,
    #     help='local proc number')
    # parser.add_argument(
    #     '--total-worker', 
    #     type=int, 
    #     help='total-worker = world_size-2')
    parser.add_argument(
        '--IP',
        default='localhost',
        help='set rpc master server IP')
    parser.add_argument(
        '--port',
        default='29513',
        help='set rpc master port')
    # parser.add_argument(
    #     '--mode',
    #     type=int, 
    #     help='set mode: 0-trainer and model, 1-env, corresponding MODE_LIST')
    args = parser.parse_args()
    
    args.env_list = []
    for i in range(args.node_num):
        args.env_list.extend(NODES[i])
    # if args.node_index == 0:
    #     assert args.total_worker == len(args.env_list)
    # else:
    #     assert args.local_proc > 0 
    args.world_size = len(args.env_list) + 2
    
    # args.mode = MODE_LIST[args.mode]
    args.device = 'cuda:%s' % (args.gpu) if args.gpu >= 0 else 'cpu'
    # args.total_env_num = args.node_num * args.env_num 
    return args

def master_proc(args):
    #=====Get args=====
    device = args.device
    model_path, load_path, debug_log_info = Config(args.load).get_config_data()
    
    env_num = len(args.env_list)
    env_ids = args.env_list
    RLrpc = rpc_rl.MasterRPC(env_num, env_ids)
    #=====Init env and model=====
    envs_rpc = RLrpc.make_vec_envs(env_ids, debug_log_info)  
    worker_model_rpc = RLrpc.make_worker_models_env_local()
    trainer = Trainer(envs_rpc, worker_model_rpc, device, model_path, debug_log_info, load_path)
    print('Init ok!')
    #=====Set worker to env =====
    shared_buffer = trainer.get_share_buf()
    RLrpc.set_model_env_local(shared_buffer)
    print('Set data Ok!')
    
    #=====start train! =====
    envs_rpc.envs_rollout()
    try:
        print('master start train!')
        trainer.train()
    except KeyboardInterrupt:
        print('Master thread get keyboard interrput!')
    except Exception as e:
        print(e)
    finally:
        envs_rpc.env_close()
        trainer.close()
        
def run_worker(rank, world_size, args):
    r"""
    This is the entry point for all processes. The rank 0 is the trainer. All
    other ranks are envs.
    """
    #In func rpc_rl.init, rank is relative proc_id to itself mode type 
    #such as, trainer 0, model 0, env 1. So the rank may be same.
    #In rpc_local, rank is absolute proc_id for distinguish mode
    try:
        #node-0 is the master.
        if args.node_index == 0:
            if rank < 2:
                if rank == 0:
                # rpc_rank 0 is the tcp server proc, and it must go to func shutdown  
                #to wait others for communicating otherwise the distirbuted program
                #cantt startup.
                    print('TCP Server start!')
                    rpc_rl.init(0, TRAINER_NAME, world_size, args.IP, args.port, args.tun)
                elif rank == 1:
                    # print('Trainer Mode start!')
                    rpc_rl.init(1, TRAINER_NAME, world_size, args.IP, args.port, args.tun)
                    master_proc(args)
            else:
                proc_id = rank - 2
                rpc_rl.init(proc_id, ENV_NAME, world_size, args.IP, args.port, args.tun, rpc_start_index=2)
        else:#woker
            # other ranks are the observer
            # proc_id = assign_env_id(args.node_index, args.node_num, rank, args.env_num)
            #args.node_index + rank * args.node_num
            start_index = 0
            for i in range(args.node_index):
                start_index += len(NODES[i])
            proc_id = start_index + rank
            rpc_rl.init(proc_id, ENV_NAME, world_size, args.IP, args.port, args.tun, rpc_start_index=2)
            # worker passively waiting for instructions from master      
    finally:
        rpc_rl.wait_util_shutdown()
        print('rpc shutdonw! ok!')

def init_prepare():
    pass

def main(args):
    # assert args.total_env_num > 0
    assert args.node_index < args.node_num
    # assert args.mode in [0, 1]
    world_size = args.world_size
    
    set_ip_forwarding()
    init_prepare()
    
    print('Master IP: %s, port: %s, tun: %s' % (args.IP, args.port, args.tun))
    # print('Mode: ', args.mode)
    print('Env num: %s' %(world_size-2))
    if args.node_index == 0:
        mp.spawn(
            run_worker,
            args=(world_size,  args, ),
            nprocs=2 + len(NODES[0]),
            join=True
        )
    else:
        mp.spawn(
            run_worker,
            args=(world_size,  args, ),
            nprocs=len(NODES[args.node_index]),
            join=True
        )
        
if __name__ == '__main__':
    args = get_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print('Main get keyboard interrput!')
    finally:
        pkill()
        print('Kill main!')