import glob
import os
import sys
import subprocess

import torch
import torch.nn as nn

from datetime import datetime


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_rnn_orthogonal(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

def assign_env_id(node_index, node_num, rank, env_num):
    # total = node_num * env_num
    if rank % 2 == 0:
        idx = node_index + rank * node_num
    else:
        # reverse for load balance 
        idx = (node_num - node_index -1) + rank * node_num
    return idx

def get_now_time_str():
    strtime = datetime.now().strftime('%Y/%m/%d-%H:%M:%Ss')
    return strtime

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
            
def pkill():
    from .game.indigo_env.env import project_root
    from subprocess import call
    path_py = os.path.join(project_root.DIR,'env','run_receiver.py')
    kill_cmds = ["pkill -f '%s'" % ('python ' + path_py),
                 "pkill -f '%s'" % ('python -c from multiprocessing.spawn import spawn_main')]
    for cmd in kill_cmds:
        sys.stderr.write('$ %s\n' % cmd)
        call(cmd, shell=True)

        

def get_network_iface(tun=False):
    if not tun:
        import socket, re
        NIC_nameindex_list = socket.if_nameindex()
        pattern = r'e[a-z]{1,}[0-9]{1,}\w{0,}'
        flag, dev = None, None
        for _, dev in NIC_nameindex_list:
            flag = re.match(pattern, dev)
            if flag is not None:
                break
    else:
        dev = 'tun0'
    return dev

def model_to_cpu(net_state_dict):
    cpu_dict = {}
    for name, value in net_state_dict.items():
        cpu_dict[name] = value.to('cpu')
    return cpu_dict
    

# ========================================================================
#                         proc
# ========================================================================
def print_cmd(cmd):
    if isinstance(cmd, list):
        cmd_to_print = ' '.join(cmd).strip()
    elif isinstance(cmd, str):
        cmd_to_print = cmd.strip()
    else:
        cmd_to_print = ''

    if cmd_to_print:
        sys.stderr.write('$ %s\n' % cmd_to_print)


def call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.check_call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.check_output(cmd, **kwargs)


def Popen(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.Popen(cmd, **kwargs)

def set_ip_forwarding():
    curr_ip_forwarding = check_output('sysctl net.ipv4.ip_forward', shell=True).decode()
    curr_ip_forwarding = curr_ip_forwarding.split('=')[-1].strip()

    if curr_ip_forwarding != '1':
        check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
        # sys.stderr.write('Changed default_qdisc from %s to %s\n'