import numpy as np
import math
from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np

class CurriculumSokobanEnv_v2(SokobanEnv):
    def __init__(self, data_path, max_steps=120):
        super(CurriculumSokobanEnv_v2, self).__init__(reset=False)
        
        self.max_steps = max_steps
        self.data_path = data_path

    def reset(self):

        if not os.path.exists(self.data_path):
            raise ValueError("You didn't generate data yet, please generate data or change the data path.")

        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.dim_room = np.array(self.room_state).shape

        starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                elif e == ' ':
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}
        num_boxes = len(boxes)

        return np.array(room_fixed), np.array(room_state), box_mapping, num_boxes

    def select_room(self):

        current_map = []
        
        if os.path.isfile(self.data_path):
            selected_map_file = self.data_path
            self.selected_map = self.data_path
            with open(self.data_path, 'r') as sf:
                for line in sf.readlines():
                    current_map.append(line)
        elif os.path.isdir(self.data_path):
            map_files = os.listdir(self.data_path)
            selected_map_file = random.choice(map_files)
            self.selected_map = selected_map_file
            with open(self.data_path + '/' + selected_map_file, 'r') as sf:
                for line in sf.readlines():
                    current_map.append(line)
        
        selected_map = current_map

        self.room_fixed, self.room_state, self.box_mapping , self.num_boxes = self.generate_room(selected_map)
