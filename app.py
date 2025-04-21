from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import numpy as np
import requests
import time
import pandas as pd
from collections import Counter
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions


import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
import tank_env


gym.register(
    id="gymnasium_env/TankEnv-v0",
    entry_point=tank_env.TankEnv,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Episode:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, env, now_state):
        self.states = []
        self.actions = []
        self.actions_log_probability = []
        self.values = []
        self.rewards = []
        self.done = False
        self.episode_reward = 0
        self.state, self.info = env.reset(options=now_state) # state는 np.array / info는 딕셔너리

class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions * 2)
        self.layer2 = nn.Linear(hidden_dimensions * 2, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        # print(f'layer 2 output shape: {x.shape}')
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        # print(f'layer 3 output shape: {x.shape}')
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

def create_agent(env, hidden_dimensions, dropout):
    INPUT_FEATURES = env.observation_space.shape[0]
    HIDDEN_DIMENSIONS = hidden_dimensions
    ACTOR_OUTPUT_FEATURES = env.action_space.n
    CRITIC_OUTPUT_FEATURES = 1
    DROPOUT = dropout
    actor = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT).to(device)
    critic = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT).to(device)
    agent = ActorCritic(actor, critic)
    return agent

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns).to(device)
    # normalize the return
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    # Normalize the advantage
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages):
    advantages = advantages.detach()
    policy_ratio = (
            actions_log_probability_new - actions_log_probability_old
            ).exp()
    # print(f'advantages = {advantages} / shape: {advantages.shape}')
    # print(f'policy_ratio = {policy_ratio} / shape: {policy_ratio.shape}')
    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(
            policy_ratio, min=1.0-epsilon, max=1.0+epsilon
            ) * advantages
    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss

def calculate_losses(surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).mean() # 딥러닝 모델은 손실을 '최소화'하려 하므로 '최대화'되어야 할 값에 음수를 취함 
    value_loss = f.smooth_l1_loss(returns, value_pred).mean() # smooth_l1_loss는 이상치에 덜 민감함

    # total_loss =  -entropy_bonus - policy_loss + value_loss
    return policy_loss, value_loss

def forward_pass(env, agent, episode, grid_epsilon, episode_count, random_episode):
    # 환경 초기화로 상태를 받아옴
    episode = episode
    agent.train() # nn.Module 클래스의 함수 
    state = torch.FloatTensor(episode.state).unsqueeze(0).to(device).clone() # 상태를 텐서로
    episode.states.append(state) # states 버퍼에 상태를 추가
    action_pred, value_pred = agent(state) # forward()가 수행된다고 봐야
    # print(f'action pred shape: {action_pred.shape} / value pred shape: {value_pred.shape}')
    if episode_count < random_episode:
        random_pred = np.array(np.array(np.random.random(4)))
        action_prob = f.softmax(torch.FloatTensor(random_pred).unsqueeze(0).to(device).clone(), dim=-1)
        # print('random prob:', action_prob, action_prob.shape)
    else:    
        action_prob = f.softmax(action_pred, dim=-1) # action_pred는 로짓(logit)이기 때문에 소프트맥스 함수로 확률로 변환
        # print('action prob:', action_prob, action_prob.shape)
    noise = torch.rand_like(action_prob) * grid_epsilon # 확률에 잡음 추가
    action_prob = action_prob + noise
    action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
    dist = distributions.Categorical(action_prob) # 확률 분포로 바꾸는데 카테고리컬하게
    if np.random.random() < grid_epsilon:  # grid epsilon 확률로 랜덤행동
        action = torch.tensor(np.random.randint(0, env.action_space.n)).unsqueeze(0).to(device).clone()
        # print(action, action.shape, 'random')
    else:
        action = dist.sample() # 확률 분포에 따라 행동 샘플링
        # print(action, action.shape, 'dist')
    log_prob_action = dist.log_prob(action) # 선택된 행동의 로그 확률을 계산하여 추후 대리 목적 함수 계산에 사용 
    state, reward, terminated, truncated, info = env.step([episode.state, episode.actions]) # 행동 텐서를 스칼라로 변환하여 환경 1스텝 진행 및 다음 값 받음
    done = terminated or truncated
    # print(f'state: {state} / type: {type(state)} / shape: {state.shape}')
    # print(f'action: {action} / type: {type(action)} / shape: {action.shape}')
    # print(f'log_prob_action: {log_prob_action} / type: {type(log_prob_action)} / shape: {log_prob_action.shape}')
    # print(f'value_pred: {value_pred} / type: {type(value_pred)} / shape: {value_pred.shape}')
    # print(f'reward: {reward} / type: {type(reward)} / shape: {reward.shape}')

    episode.actions.append(action) # actions 버퍼에 행동을 추가
    episode.actions_log_probability.append(log_prob_action) # 마찬가지로 버퍼에 추가
    episode.values.append(value_pred)
    episode.rewards.append(reward)
    # print(f'Episode Info // Value: {value_pred.shape} // Reward: {reward.shape}')
    episode.done = done
    episode.episode_reward += reward
    episode.info = info
    return action.item(), done, info

def update_policy(
        agent,
        episode,
        discount_factor,
        optimizer,
        ppo_steps,
        epsilon,
        entropy_coefficient,
        batch_size,
        value_loss_coef
        ):
    states = torch.cat(episode.states).to(device) # 리스트에 저장된 상태 텐서들을 연결하여 단일 텐서로 만듬
    actions = torch.cat(episode.actions).to(device)
    actions_log_probability_old = torch.cat(episode.actions_log_probability).to(device)
    values = torch.cat(episode.values).squeeze(-1).to(device).clone() # 가치 예측 텐서들을 연결하고 마지막 차원을 제거
    returns = calculate_returns(episode.rewards, discount_factor)
    # print(f'returns: {returns} / shape: {returns.shape}')
    advantages = calculate_advantages(returns, values)
    # print(f'advantages: {advantages} / shape : {advantages.shape}')
    episode_reward = episode.episode_reward
    batch_size = batch_size
    total_policy_loss = 0
    total_value_loss = 0
    actions = actions.detach() # 과거의 행동을 고정해야 같은 행동에 대한 새로운 예측치를 산출할 수 있음 
    actions_log_probability_old = actions_log_probability_old.detach() # 대리 손실 함수 계산을 위해 과거 확률 고정
    training_results_dataset = TensorDataset(
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)
    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=batch_size,
            shuffle=True)
    for i in range(ppo_steps):
        print(f'PPO Steps {i} Start')
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # 입력된 모든 상태 정보에 대한 새로운 행동, 가치 예측치를 산출 
            action_pred, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1).to(device).clone()
            action_prob = f.softmax(action_pred, dim=-1)
            probability_distribution_new = distributions.Categorical(
                    action_prob)
            entropy = probability_distribution_new.entropy()

            # 과거 행동 확률과 새로운 행동 확률을 calculate_surrogate_loss 함수에 전달  
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            # print(f'actions_log_probability_old: {actions_log_probability_old} / shape : {actions_log_probability_old.shape}')
            # print(f'actions_log_probability_new: {actions_log_probability_new} / shape : {actions_log_probability_new.shape}')
            
            surrogate_loss = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
            # 산출된 대리 목적 손실, 엔트로피, 누적 보상과 예측치를 사용하여 정책 손실과 가치 손실을 산출 
            policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred,
                    )
            value_loss = value_loss_coef * value_loss
            # total_loss = policy_loss + value_loss * value_loss_coef
            # 최적화 함수 기울기 초기화
            optimizer.zero_grad()
            # 손실 기울기 역전파
            # total_loss.backward()
            policy_loss.backward()
            value_loss.backward()
            # 기울기 합산하여 적용 후 다음 단계로 진행
            optimizer.step()
            time.sleep(0.2)
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    # 평균 정책 손실과 평균 가치 손실을 반환 
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def send_reset_message():
    requests.get('http://172.21.160.1:5522/click', timeout=10)


def change_degree(my_d, x, y, des_x, des_y):
    if my_d > 180:
        direction = -(360-my_d)
    else:
        direction = my_d
    atan = math.atan2(des_x- x,des_y -y)
    azimuth = math.degrees(atan)
    heading = round(azimuth - direction, 2)
    if heading > 180:
        heading = -(heading - 180)
    if heading < -180:
        heading = 360 + heading
    sin = np.sin(heading)
    cos = np.cos(heading)
    return sin, cos

# Initialize Flask server
app = Flask(__name__)
log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
# log.disabled = False


# Load YOLO model
model = YOLO('yolov8n.pt')

# Action commands
action_command = ["FIRE"]

MAX_EPISODES = 200              
DISCOUNT_FACTOR = 0.99
REWARD_THRESHOLD = 8
PRINT_INTERVAL = 10
PPO_STEPS = 8
N_TRIALS = 512
EPSILON = 0.2
ENTROPY_COEFFICIENT = 0.1
HIDDEN_DIMENSIONS = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
VALUE_LOSS_COEF = 0.5

env = gym.make('gymnasium_env/TankEnv-v0', max_steps=N_TRIALS, threshold = REWARD_THRESHOLD)
rng = np.random.default_rng(seed=8)

# 모델 초기화 // 모델 웨이트 입혀서 수행 가능.
agent = create_agent(env, hidden_dimensions=HIDDEN_DIMENSIONS, dropout=DROPOUT)
# checkpoint = torch.load("model_weights_20.pth", map_location=device)
# agent.load_state_dict(checkpoint)
print('Agent Initialized')

# 코드 작성 필요 

optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
print('Optimizer Initialized')

step_counter = 0
now_state = [0, 0, 0, 0, 0, 0, 0]
episode = None

# 학습 기록을 위한 카운터 변경
# 30
episode_counter = 0

move_command = ""
episode_rewards = []
episode_actions = []
command_list = []
episode_commands = []

calculating = False

action_to_direction = {
            0: "W",
            1: "S",
            2: "A",
            3: "D",
            # 4: 'STOP'
        }

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')

    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)  # Save temporary image

    # Perform detection
    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes

    # Filter only specific object classes
    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })

    return jsonify(filtered_results)


@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)

    global agent
    global now_state
    global episode
    global move_command
    global episode_counter
    global step_counter
    global episode_rewards
    global command_list
    global episode_commands
    global calculating

    if not data:
        return jsonify({"error": "No JSON received"}), 400
    
    if calculating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    
    player_x = round(data['playerPos']['x'], 2)
    player_y = round(data['playerPos']['z'], 2)
    des_x = round(data['enemyPos']['x'], 2)
    des_y = round(data['enemyPos']['z'], 2)
    player_speed = round(data['playerSpeed'], 4)
    sin, cos = change_degree(data['playerBodyX'], player_x, player_y, des_x, des_y)

    now_state[0] = player_x
    now_state[1] = player_y
    now_state[2] = des_x
    now_state[3] = des_y
    now_state[4] = player_speed
    now_state[5] = sin
    now_state[6] = cos
    
    if episode is None:
        calculating = True
        episode = Episode(env, now_state)
        print(f'Episode {episode_counter + 1} has been initialized')
        calculating = False
        return jsonify({"status": "success", "control": "reset"})

    if episode.done:
        # try:
        calculating = True
        print(f'Episode {episode_counter + 1} has been finished!!')
        policy_loss, value_loss = update_policy(agent, episode, discount_factor=DISCOUNT_FACTOR, \
                                                optimizer=optimizer, ppo_steps=PPO_STEPS, epsilon=EPSILON, \
                                                entropy_coefficient=ENTROPY_COEFFICIENT, batch_size=BATCH_SIZE, value_loss_coef=VALUE_LOSS_COEF)
        print(f'Policy Loss: {policy_loss}, Value Loss: {value_loss}')
        
        episode_rewards.append(episode.rewards)
        episode = None
        step_counter = 0
        now_state = [0, 0, 0, 0, 0, 0, 0]
        episode_commands.append(command_list)
        counter = Counter(command_list)
        print(counter)
        command_list = []
        move_command = ""

        episode_counter += 1

        try:
            if episode_counter % PRINT_INTERVAL == 0:
                torch.save(agent.state_dict(), f'model_weights_{episode_counter}.pth')
            torch.save(agent.state_dict(), f'model_weights.pth')
            seriese = pd.DataFrame(episode_rewards)
            seriese.to_csv(f'./rewards.csv', encoding='cp949')
            commands = pd.DataFrame(episode_commands)
            commands.to_csv(f'./commands.csv', encoding='cp949')
        except Exception() as e:
            print(f'Error has been occurred during saving logs. {e}')

        calculating = False
        print('Ready for the next episode!')
        time.sleep(1)
        return jsonify({"status": "success", "control": "reset"})
        # except Exception as e:
        #     calculating = False
        #     episode = None
        #     step_counter = 0
        #     command_list = []
        #     return jsonify({"status": "ERROR", "message": str(e)}), 400


    else:
        calculating = True
        episode.state = now_state
        move_command, done, info = forward_pass(env, agent, episode=episode, grid_epsilon = ENTROPY_COEFFICIENT * (1 - episode_counter/MAX_EPISODES), episode_count = episode_counter, random_episode=10)
        move_command = action_to_direction[move_command]
        command_list.append(move_command)
        episode.done = done
        info_reward = info['reward']
        info_distance = info['distance']
        print(f'Episode {episode_counter + 1} / Step {step_counter} has been finished / state: {episode.state[:4]} {episode.state[5]:.2f} {episode.state[6]:.2f} / reward: {info_reward} / distance: {info_distance}' )
        if step_counter % 10 == 0 :
            # print(command_list)
            print(Counter(command_list))
        step_counter += 1
        calculating = False
        return jsonify({"status": "success", "message": "Data received"}), 200


   



@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()

    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (int(x), int(z))  # Ignore height (y)
        # print(f"Updated Position: {current_position}")
        return jsonify({"status": "OK", "current_position": current_position})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400


@app.route('/get_move', methods=['GET'])
def get_move():
    global step_counter
    # global calculating
    # if calculating:
    #     return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    global move_command
    if move_command:
        command = move_command
        print(f'Sent Move Command: {command} / Step: {step_counter}')
        return jsonify({'move': command, 'weight': 1})
    else:
        return jsonify({"move": "STOP"})


@app.route('/get_action', methods=['GET'])
def get_action():
    if calculating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    global action_command

    if action_command:
        command = action_command.pop(0)
        print(f"Sent Action Command: {command}")
        return jsonify({"turret": command})
    else:
        return jsonify({"turret": " "})


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()

    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()

    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x_dest, y_dest, z_dest = map(float, data["destination"].split(","))
        # print(f"Received destination: x={x_dest}, y={y_dest}, z={z_dest}")

        print(f'Now state: {now_state}')

        return jsonify({
            "status": "OK",
            "destination": {
                "x": x_dest,
                "y": y_dest,
                "z": z_dest
            }
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    obstacle_data = request.get_json()

    if not obstacle_data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    # print("Received obstacle data:", obstacle_data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'}), 200

@app.route('/init', methods=['GET'])
def init():
    global rng
    global calculating
    global episode_counter
    if calculating:
        return jsonify({"status": "Calculating", "message": "Some Calculation are going on..."}), 102
    calculating = True
    curriculum = 40 + episode_counter
    while True:
        random_coord = rng.integers(low=10, high=290, size=4)
        x = int(random_coord[0])
        z = int(random_coord[1])
        if episode_counter < 100:
            des_x = rng.integers(low= x - curriculum, high= x + curriculum, size = 1)[0]
            des_z = rng.integers(low= z - curriculum, high= z + curriculum, size = 1)[0]
        else:
            des_x = int(random_coord[2])
            des_z = int(random_coord[3])
        distance = np.sqrt((x - des_x) ** 2 + (z - des_z) ** 2)
        if (distance > REWARD_THRESHOLD + 10) and (des_x > 5 and des_x < 295 and des_z > 5 and des_z < 295):
            break
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": int(x),  #Blue Start Position
        "blStartY": 10,
        "blStartZ": int(z),
        "rdStartX": int(des_x), #Red Start Position
        "rdStartY": 10,
        "rdStartZ": int(des_z),
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False
    }
    print("🛠️ Initialization config sent via /init:", config["blStartX"], config["blStartZ"], config["rdStartX"], config["rdStartZ"])
    calculating = False
    # send_reset_message()
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5256, debug=True)
