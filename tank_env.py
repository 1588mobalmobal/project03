import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000, threshold = 3.0 , test_mode=False):
        super().__init__() 

        self.action_space = gym.spaces.Discrete(4)
        
        self.observation_space = gym.spaces.Box(
            low = -300, high=300, shape=(6,), dtype=np.float32
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
        x, y, dest_x, dest_y, speed, direction = map(float, data)
        self.prev_distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)
        state = np.array([x / 300, y / 300, dest_x / 300, dest_y / 300, speed / 100, direction / 180], dtype=np.float32)
        self.current_state = state      

        info = {'distance': self.prev_distance, 'raw': options}

        print(f'Env Reset / info : {info}')

        return state, info
    
    def step(self, sim_data):
        self.steps += 1
        data = sim_data[0]
        actions = sim_data[1]

        x, y, dest_x, dest_y, speed, direction = map(float, data)
        
        # at_boundary = x < 1 / 300 or x > 299 or y < 1 or y > 299
        distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)
        
        phi = -distance / 300 * 2
        prev_phi = -self.prev_distance / 300 * 2
        reward = phi - prev_phi
        reward -= 0.002 # 시간 패널티
        # reward -= 0.02  # 시간 패널티
        # reward += 0.1 * (1 - (distance + 0.001) / 300)  # 절대 거리 보너스 (300은 환경 크기)
        # if at_boundary and (distance - self.prev_distance > - 1):
        #     reward -= 0.2  # 경계 패널티
        # if distance < self.prev_distance:
        #     reward += 0.05  # 거리 감소 보너스
        if len(actions) > 5 and actions[-5:] == [2, 2, 2, 2, 2]:
            reward -= 0.3
        if len(actions) > 5 and actions[-5:] == [3, 3, 3, 3, 3]:
            reward -= 0.3
        
        terminated = False
        truncated = False

        if distance < self.threshold:
            terminated = True
            self.steps = 0
            reward += 10

        if self.steps >= self.max_steps:
            truncated = True
            self.steps = 0

        state = np.array([x / 300, y / 300, dest_x / 300, dest_y / 300, speed / 100, direction / 180], dtype=np.float32)

        self.current_state = state

        info = {'distance': round(distance - self.prev_distance, 4), 'reward': round(reward, 4)}

        self.prev_distance = distance

        return state, reward, terminated, truncated, info
