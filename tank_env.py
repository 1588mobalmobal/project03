import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000, threshold = 3.0 , test_mode=False):
        super().__init__() 

        self.action_space = gym.spaces.Discrete(4)
        
        self.observation_space = gym.spaces.Box(
            low = 0, high=300, shape=(4,), dtype=np.float32
        )
        self.steps = 0
        self.max_steps = max_steps
        self.test_mode = test_mode

        self.threshold = threshold
        self.prev_distance = None
        self.current_state = None
        
        self.action_to_direction = {
            0: 'W',
            1: 'S',
            2: 'A',
            3: 'D',
            # 4: 'STOP'
        }

        print('Tank Env initialized')
        

    def reset(self, seed=None, options=None):
        # 시뮬레이터 리셋 로직 구현 필요
        data = options
        x, y, dest_x, dest_y = map(float, data)
        state = np.array([x, y, dest_x, dest_y], dtype=np.float32)

        self.current_state = state
        self.prev_distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)

        info = {'distance': self.prev_distance, 'raw': options}

        print(f'Env Reset / info : {info}')

        return state, info
    
    def step(self, sim_data):
        self.steps += 1
        data = sim_data
        x, y, dest_x, dest_y = map(float, data)
        state = np.array([x, y, dest_x, dest_y], dtype=np.float32)
        distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)
        reward = (self.prev_distance - distance) / 100
        # if distance < self.threshold:
        #     reward += 100
        reward -= 0.01
        self.prev_distance = distance

        terminated = False
        truncated = False

        if distance < self.threshold:
            terminated = True
            self.steps = 0

        if self.steps > self.max_steps:
            truncated = True
            self.steps = 0

        self.current_state = state

        info = {'distance': self.prev_distance, 'raw': sim_data}

        if self.steps % 10 == 0:
            print(f'step {self.steps} completed / info: {info} / reward: {reward}')

        return state, reward, terminated, truncated, info
