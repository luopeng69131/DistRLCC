import os
import sys
from os import path
DIR = path.abspath(path.join(path.dirname(path.abspath(__file__)), os.pardir))
model_module_path = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../../..'))
src_model_module_path = path.abspath(path.join(model_module_path, '..'))
sys.path.append(DIR)
