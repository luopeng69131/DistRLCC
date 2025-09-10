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
import project_root
import sys
sys.path.extend([project_root.model_module_path, project_root.src_model_module_path])

from os import path
from env.sender import Sender
from model import TestModel
from helpers.helpers import normalize, one_hot, softmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    sender = Sender(args.port, debug=args.debug)
    
    model_path = path.join(project_root.DIR, 'dagger', 'trained_model')
    test_model = TestModel(
        [Sender.obs_dim,
         Sender.action_cnt,
         Sender.max_steps],
        model_path
        )
    
    sender.set_model(test_model)
    try:
        sender.reset()
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
