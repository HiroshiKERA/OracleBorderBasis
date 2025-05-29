import os
import re
import sys
import math
import time
import pickle

import errno
import signal
from functools import wraps, partial
import torch
# from utils.logger import create_logger

CUDA = True

def count_cuda_devices():
    # Get CUDA_VISIBLE_DEVICES from environment variables
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if cuda_visible_devices is not None:
        # If CUDA_VISIBLE_DEVICES is set, count by splitting with commas
        visible_devices = cuda_visible_devices.split(',')
        return len(visible_devices)
    else:
        # If CUDA_VISIBLE_DEVICES is not set, count all available devices
        return torch.cuda.device_count()
    
def write_to_file(filename, string_list):
    with open(filename, "w") as file:
        for string in string_list:
            file.write(string + "\n")

def to_cuda(tensor_dict):
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].cuda()
            
    return tensor_dict


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator

def read_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_file_split(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return [line.strip().split(':') for line in lines]   # each line separated as "source : target"


def convert_sage_types(obj):
    """
    Convert SageMath types to native Python types recursively.
    
    Args:
        obj: Any object that might contain Sage types
        
    Returns:
        The same object structure with Sage types converted to Python types
    """
    # For dictionaries, convert each value recursively
    if isinstance(obj, dict):
        return {key: convert_sage_types(value) for key, value in obj.items()}
    
    # For lists, convert each element recursively
    if isinstance(obj, list):
        return [convert_sage_types(item) for item in obj]
    
    # For tuples, convert each element recursively
    if isinstance(obj, tuple):
        return tuple(convert_sage_types(item) for item in obj)
    
    # For Sage Integer type, convert to Python int type
    if str(type(obj).__module__).startswith('sage'):
        try:
            return int(obj)
        except:
            try:
                return float(obj)
            except:
                return str(obj)
    
    # For other types, return as is
    return obj