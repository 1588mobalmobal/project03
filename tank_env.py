import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

class TankEnv(gym.Env):
    def __init__(self, max_steps = 1000, threshold = 3.0 , test_mode=False):
        super().__init__() 

        self.action_space = gym.spaces.Discrete(4)
        
        self.observation_space = gym.spaces.Box(
            low = -300, high=300, shape=(7,), dtype=np.float32
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
        x, y, dest_x, dest_y, speed, sin, cos = map(float, data)
        self.prev_distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)
        state = np.array([x / 300, y / 300, dest_x / 300, dest_y / 300, speed / 100, sin, cos], dtype=np.float32)
        self.current_state = state

        info = {'distance': self.prev_distance, 'raw': options}

        print(f'Env Reset / info : {info}')

        return state, info
    
    def step(self, sim_data):
        self.steps += 1
        data = sim_data[0]
        actions = sim_data[1]

        x, y, dest_x, dest_y, speed, sin, cos = map(float, data)
        
        at_boundary = x < 1 or x > 299 or y < 1 or y > 299
        distance = np.sqrt((x - dest_x) ** 2 + (y - dest_y) ** 2)
        reward = (self.prev_distance - distance) / 100  # 기본 보상 강화
        reward -= 0.01  # 시간 패널티 증가
        reward += 0.01 * (1 - distance / 300)  # 절대 거리 보너스 축소

        stuck = False

        last_move = actions[-1:]
        if len(actions) > 7 and actions[-7:] == last_move * 7:
            reward -= 0.03  # 반복 행동 패널티 강화
        if abs(distance - self.prev_distance) < 0.25:
            reward -= 0.02  # 정체 패널티
            stuck = True

        if at_boundary:
            reward -= (distance) / 500
            if stuck:
                reward -= 0.1
        if at_boundary:
            if (x < 1 and last_move == 'E') or (x > 299 and last_move == 'W') or (y < 1 and last_move == 'N') or (y > 299 and last_move == 'S'):
                reward += 0.02

        terminated = False
        truncated = False

        if distance < 24:  # 근거리 보너스
            reward += 0.1
        if distance < self.threshold:
            terminated = True
            reward += 5
            self.steps = 0
            print('😊😊😊😊 Touched the Goal 😊😊😊😊')

        if self.steps >= self.max_steps:
            truncated = True
            self.steps = 0
            print('📌📌📌📌 Out of max steps 📌📌📌📌')

        state = np.array([x / 300, y / 300, dest_x / 300, dest_y / 300, speed / 100, sin, cos], dtype=np.float32)

        self.current_state = state

        info = {'distance': round(distance, 4), 'reward': round(reward, 4)}

        self.prev_distance = distance

        return state, reward, terminated, truncated, info
