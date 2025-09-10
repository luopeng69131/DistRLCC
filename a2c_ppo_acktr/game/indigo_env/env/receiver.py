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


import sys
import json
import socket
import select
import datagram_pb2
import project_root
from helpers.helpers import READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, ALL_FLAGS


class Receiver(object):
    def __init__(self, ip, port, env_id=-1, debug=False):
        self.peer_addr = (ip, port)
        self.env_id = env_id
        self.debug = debug
        
        self.port = None
        self.handshake_first_flag = False
        
        self.init_conn()

    def cleanup(self):
        self.sock.close()
        
    def init_conn(self):
        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        if self.port is None:
            self.port = self.sock.getsockname()[1]
        else:
            self.sock.bind(('0.0.0.0', self.port))
        # sys.stderr.write('[receiver %s] Listening on port %s\n' %
        #                  (self.env_id, self.sock.getsockname()[1]))
        
        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def construct_ack_from_data(self, serialized_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        data = datagram_pb2.Data()
        data.ParseFromString(serialized_data)

        ack = datagram_pb2.Ack()
        ack.seq_num = data.seq_num
        ack.send_ts = data.send_ts
        ack.sent_bytes = data.sent_bytes
        ack.delivered_time = data.delivered_time
        ack.delivered = data.delivered
        ack.ack_bytes = len(serialized_data)

        return ack.SerializeToString()
    
    def wavehand(self):
        """Wavehand with peer sender. Must be called after run()."""
        
        self._retransmit('Bye from receiver', 'Bye ack from sender')
        # sys.stderr.write('[receiver %s] Wavehand OK\n' % (self.env_id))
    
    def _retransmit(self, info, wait_info):
        TIMEOUT = 1000  # ms
        retry_times = 0
        wait_info_bytes = wait_info.encode() if wait_info is not None else None
            
        retransmit_flag = True
        while True:
            if retransmit_flag:
                self.sock.sendto(info.encode(), self.peer_addr)
            events = self.poller.poll(TIMEOUT)

            if not events:  # retransmission
                retransmit_flag = True
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write('[receiver %s] Retransmission failed after 10 retries\n' % (self.env_id))
                    if wait_info is not None:
                        raise OSError('[receiver %s] Retransmission fails' % (self.env_id))
                    else:
                        sys.stderr.write('[receiver %s] Retransmission failed but info is None so return to run\n' % (self.env_id))
                        return
                else:
                    if retry_times == 1:
                        sys.stderr.write(
                            '[receiver %s] Retransmission timed out and retrying...\n' % (self.env_id))
                    continue
            else:
                retransmit_flag = False #not bye pks close retransmission
                for fd, flag in events:
                    assert self.sock.fileno() == fd

                    if flag & ERR_FLAGS:
                        sys.exit('Channel closed or error occurred')

                    if flag & READ_FLAGS:
                        msg, addr = self.sock.recvfrom(1600)
                        # msg = msg.decode()
                        if addr == self.peer_addr:
                            if wait_info is None:
                                return 
                            elif msg[0] == wait_info_bytes[0] and msg[-1] == wait_info_bytes[-1]:
                                if msg.decode() == wait_info:
                                    return 
                            else:
                                continue
    
    def handshake(self):
        if not self.handshake_first_flag:
            self._handshake_first()
            self.handshake_first_flag = True
        else:
            self._handshake_others()
        
    def _handshake_others(self):
        """Handshake with peer sender. Must be called before run()."""
        if self.debug:
            sys.stderr.write('[receiver %s] handshake_others start\n' % (self.env_id))
        self.sock.setblocking(0)  # non-blocking UDP socket
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        
        recv_hello_flag = False
        while(recv_hello_flag is False):
            events = self.poller.poll(None)
            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1600)
                    msg = msg.decode()

                    # sys.stderr.write('[receiver %s] handshake_others msg: %s\n' % (self.env_id, msg[:4]))
                    if addr == self.peer_addr:
                        if msg == 'Hello from sender':
                            recv_hello_flag = True
                            if self.env_id == 0:
                                sys.stderr.write('[receiver %s] handshake_others msg: %s\n' % (self.env_id, msg[:4]))
                            break
                        
        self._retransmit('Hello from receiver', None)
        # sys.stderr.write('[receiver %s] handshake_others ok\n' % (self.env_id))
        #None means any info can help it leave this func
        
                        
    def _handshake_first(self):
        """Handshake with peer sender. Must be called before run()."""
        # if self.debug:
        #     sys.stderr.write('[receiver %s] handshake_first start\n' % (self.env_id))
        self.sock.setblocking(0)  # non-blocking UDP socket
        TIMEOUT = 1000  # ms
        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            self.sock.sendto('Hello from receiver'.encode(), self.peer_addr)
            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write('[receiver %s] Handshake failed after 10 retries\n' % (self.env_id))
                    raise OSError('[receiver %s] Handshake failed' % (self.env_id))
                else:
                    if retry_times == 1:
                        sys.stderr.write('[receiver %s] Handshake timed out and retrying...\n' % (self.env_id))
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1600)
                    # print(msg)
                    msg = msg.decode()

                    if addr == self.peer_addr:
                        if msg == 'Hello from sender':
                            sys.stderr.write('[receiver %s] handshake OK\n' % (self.env_id))
                            # self.handshake_first_flag = True
                            return
                        else:
                            # 'Hello from sender' was presumably lost
                            # received subsequent data from peer sender
                            sys.stderr.write('[receiver %s] handshake meet old pks\n' % (self.env_id))
                            ack = self.construct_ack_from_data(msg)
                            if ack is not None:
                                self.sock.sendto(ack, self.peer_addr)
                            # self.handshake_first_flag = True
                            return

    def run(self):
        self.sock.setblocking(1)  # blocking UDP socket

        while True:
            serialized_data, addr = self.sock.recvfrom(1600)
            # serialized_data = serialized_data
            
            if addr == self.peer_addr:
                if serialized_data[0] == 66 and serialized_data[-1] == 114:# 'Bye from sender'
                    #recv wave pkt 
                    serialized_data = serialized_data.decode()
                    if serialized_data == 'Bye from sender':
                        if self.env_id == 0:
                            sys.stderr.write('[receiver %s] get end syn!\n' % (self.env_id))
                        break
                
                ack = self.construct_ack_from_data(serialized_data)
                if ack is not None:
                    self.sock.sendto(ack, self.peer_addr)
