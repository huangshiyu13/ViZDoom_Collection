"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: rl_utils.py
"""

import random

class Decay():
    def __init__(self, decay_end_step, decay_start_rate=0.9, decay_end_rate=0.1):
        self.decay_start_rate = decay_start_rate
        self.decay_end_rate = decay_end_rate
        self.decay_end_step = decay_end_step
        self.decay_rate_now = self.decay_start_rate

    def random_choose(self, step):
        if step >= self.decay_end_step:
            self.decay_rate_now = self.decay_end_rate
            return random.random() < self.decay_end_rate

        self.decay_rate_now = (float(self.decay_end_step - step) / self.decay_end_step) * (
                    self.decay_start_rate - self.decay_end_rate) + self.decay_end_rate

        return random.random() < self.decay_rate_now

    def get_decayrate(self):
        return self.decay_rate_now


class HandplayAction():
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        while True:
            action_id = input('please enter your action id(<{}):'.format(self.action_size))
            if action_id.isdigit():
                action_id = int(action_id)
                if action_id < self.action_size and action_id >= 0:
                    return action_id
