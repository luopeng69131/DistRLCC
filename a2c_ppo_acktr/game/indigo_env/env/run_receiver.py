#!/usr/bin/env python

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


import argparse
from receiver import Receiver
import sys, time 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    # parser.add_argument('proc_id', type=int, default=-1)
    parser.add_argument('env_id', type=int, default=-1)
    parser.add_argument('debug', type=int, default=0)
    args = parser.parse_args()
    
    receiver = Receiver(args.ip, args.port, args.env_id, args.debug)
    
    try:           
        receiver.handshake()
        receiver.run()
        # receiver.wavehand()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.cleanup()


if __name__ == '__main__':
    main()
