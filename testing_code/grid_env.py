import gymnasium as gym
import pygame

from typing import Optional
import numpy as np

class GridWorldEnv(gym.Env):
    # 지원하는 렌더링 모드 정의 
    metadata = {"render_modes" : ["human", "rgb_array", "ansi"]}
    # 초기화 함수
    def __init__(self, size: int = 30, max_steps: int = 150, render_mode: Optional[str] = None):

        self.size = size
        # 에이전트와 타겟의 위치를 정의 : reset에서 무작위 선택되고 step에서 업데이트됨 
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._distance = np.array([-1,], dtype=np.int32)

        # 최대 스텝 수 제한을 위한 step
        self._steps = 0
        self.max_steps = max_steps

        # obserbation 정의. 에이전트와 타겟의 위치를 반환
        self.observation_space = gym.spaces.Dict({
            'agent' : gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            'target' : gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            'distance' : gym.spaces.Box(-(size + size), (size + size), shape=(1,), dtype=int)
        })

        # 행동 정의. 현재는 '위, 아래, 오른쪽, 왼쪽' 4개의 행동 
        self.action_space = gym.spaces.Discrete(4)
        # 이산 행동의 행동 결과를 정의
        self._action_to_direction = {
            0 : np.array([1, 0]), # 오른쪽
            1 : np.array([0, 1]), # 위
            2 : np.array([-1, 0]), # 왼쪽
            3 : np.array([0, -1]), # 아래
        }

        # pygame 렌더링 설정
        # if render_mode not in self.metadata["render_modes"]:
        #     raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode}")
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.cell_size = 30
        self.window_size = (self.size * self.cell_size, self.size * self.cell_size)


    # 에이전트와 타깃의 위치를 반환 
    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location, 'distance': self._distance}
    # 에이전트와 타깃의 거리를 반환 
    def _get_info(self):
        return {
            'distance': np.linalg.norm(self._agent_location - self._target_location, ord=1) # ord=1 : 맨해튼 거리 반환 
        }
    
    # 환경 초기화 함수
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # reset 오버라이딩
        super().reset(seed=seed)

        # 에이전트 위치를 무작위로 선택. 이때, gym.Env 클래스에 내장된 np_random 객체에 접근하여 난수 생성. np_random 객체는 환경별 난수 생성기
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # 타겟의 위치를 에이전트 위치와 동일하게 하여 다음 트리거를 확정 
        self._target_location = self._agent_location
        # 타겟 위치를 에이전트 위치와 다르게 할당 
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        observatioin = self._get_obs()
        info = self._get_info()

        return observatioin, info
    
    # 단계를 진행하는 함수 
    def step(self, action):
        # 단계 한번마다 steps 증가
        self._steps += 1
        # 행동(0, 1, 2, 3 중 하나)을 이동할 방향으로 매핑
        direction = self._action_to_direction[action]
        # np.clip을 사용해 그리드 경계를 벗어나지 못하도록 제한
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        self._distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)

        # 환경 완료 조건 명시 
        terminated = False
        truncated = False
        if np.array_equal(self._agent_location, self._target_location):
            terminated = True
            self._steps = 0
        elif self._steps > self.max_steps:
            truncated = True
            self._steps = 0
        
        # 보상 계산 ######################################################################################################
        reward = 100 if terminated else self._distance * -0.05
        # reward = -np.log(self._distance + 0.000000000000000000000000000000000000001) + 2
        
        # 관찰값 및 정보 생성 
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info    
    
    def render(self):
        if self.render_mode == "human":
            # 창 초기화
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption("GridWorldEnv")
                self.clock = pygame.time.Clock()

            # 이벤트 처리 (창 닫기)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            # 화면 지우기
            self.screen.fill((255, 255, 255))

            # 그리드 그리기
            for x in range(self.size + 1):
                pygame.draw.line(self.screen, (200, 200, 200),
                                 (x * self.cell_size, 0),
                                 (x * self.cell_size, self.window_size[1]))
            for y in range(self.size + 1):
                pygame.draw.line(self.screen, (200, 200, 200),
                                 (0, y * self.cell_size),
                                 (self.window_size[0], y * self.cell_size))
                
            # 에이전트와 타겟 그리기
            agent_pos = (self._agent_location[0] * self.cell_size + self.cell_size // 2,
                         self._agent_location[1] * self.cell_size + self.cell_size // 2)
            target_pos = (self._target_location[0] * self.cell_size + self.cell_size // 2,
                          self._target_location[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, (0, 0, 255), agent_pos, self.cell_size // 4)  # 파란색 에이전트
            pygame.draw.circle(self.screen, (255, 0, 0), target_pos, self.cell_size // 4)  # 빨간색 타겟

            # 화면 업데이트
            pygame.display.flip()
            # pygame.time.delay(5)
            self.clock.tick(20)  # FPS 30으로 제한

        elif self.render_mode == "rgb_array":
            # rgb_array 구현 (필요 시)
            pass

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


