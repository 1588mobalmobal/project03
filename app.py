from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import numpy as np
import requests
import time
import pandas as pd

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

class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
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
        self.state, self.info = env.reset(options=now_state)

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

def forward_pass(env, agent, episode, now_state, episode_counter):
    # 환경 초기화로 상태를 받아옴
    episode = episode
    agent.train() # nn.Module 클래스의 함수 
    state = torch.FloatTensor(episode.state).unsqueeze(0).to(device) # 상태를 텐서로
    episode.states.append(state) # states 버퍼에 상태를 추가
    action_pred, value_pred = agent(state) # forward()가 수행된다고 봐야  
    action_prob = f.softmax(action_pred, dim=-1) # action_pred는 로짓(logit)이기 때문에 소프트맥스 함수로 확률로 변환 
    dist = distributions.Categorical(action_prob) # 확률 분포로 바꾸는데 카테고리컬하게
    print(dist.probs)
    # if episode_counter < 10:
    #     action = np.random.randint(0, 3)
    #     action = torch.tensor(action).to(device).unsqueeze(0)
    # else:
    action = dist.sample() # 확률 분포에 따라 행동 샘플링
    log_prob_action = dist.log_prob(action) # 선택된 행동의 로그 확률을 계산하여 추후 대리 목적 함수 계산에 사용 
    state, reward, terminated, truncated, _ = env.step(now_state) # 행동 텐서를 스칼라로 변환하여 환경 1스텝 진행 및 다음 값 받음
    done = terminated or truncated
    episode.actions.append(action) # actions 버퍼에 행동을 추가
    episode.actions_log_probability.append(log_prob_action) # 마찬가지로 버퍼에 추가
    episode.values.append(value_pred)
    episode.rewards.append(reward)
    episode.done = done
    episode.episode_reward += reward
    return action.item(), done

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
    values = torch.cat(episode.values).squeeze(-1).to(device) # 가치 예측 텐서들을 연결하고 마지막 차원을 제거
    returns = calculate_returns(episode.rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
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
    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # 입력된 모든 상태 정보에 대한 새로운 행동, 가치 예측치를 산출 
            action_pred, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1).to(device)
            action_prob = f.softmax(action_pred, dim=-1)
            probability_distribution_new = distributions.Categorical(
                    action_prob)
            entropy = probability_distribution_new.entropy()
            print(f'Entropy: {entropy}')
            # 과거 행동 확률과 새로운 행동 확률을 calculate_surrogate_loss 함수에 전달  
            actions_log_probability_new = probability_distribution_new.log_prob(actions)
            print(f"Action Probs: {probability_distribution_new.probs}")
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
            total_loss = policy_loss + value_loss * value_loss_coef
            # 최적화 함수 기울기 초기화
            optimizer.zero_grad()
            # 손실 기울기 역전파
            total_loss.backward()
            # 기울기 합산하여 적용 후 다음 단계로 진행 
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    # 평균 정책 손실과 평균 가치 손실을 반환 
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def send_reset_message():
    request = requests.get('http://172.21.160.1:5522/click', timeout=10)


# Initialize Flask server
app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Action commands
action_command = ["FIRE"]

MAX_EPISODES = 500              
DISCOUNT_FACTOR = 0.99
REWARD_THRESHOLD = 8
PRINT_INTERVAL = 10
PPO_STEPS = 10
N_TRIALS = 512
EPSILON = 0.25
ENTROPY_COEFFICIENT = 0.05
HIDDEN_DIMENSIONS = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
VALUE_LOSS_COEF = 0.5

env = gym.make('gymnasium_env/TankEnv-v0', max_steps=N_TRIALS, threshold = REWARD_THRESHOLD)
rng = np.random.default_rng(seed=123)
agent = create_agent(env, hidden_dimensions=HIDDEN_DIMENSIONS, dropout=DROPOUT)
print('Agent Initialized')
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
print('Optimizer Initialized')

step_counter = 1
now_state = [0, 0, 0, 0, 0]
episode = None
episode_counter = 0
move_command = ""
destination = [0, 0]
episode_rewards = []
episode_actions = []
command_list = []
episode_commands = []

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

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # print("📨 /info data received:", data)
    now_state[0] = round(data['playerPos']['x'], 2)
    now_state[1] = round(data['playerPos']['z'], 2)
    now_state[2] = round(data['enemyPos']['x'], 2)
    now_state[3] = round(data['enemyPos']['z'], 2)
    now_state[4] = data['playerSpeed']


    if episode is None:
        episode = Episode(env, now_state)
        print(f'Episode {episode_counter + 1} has been initialized')

    if episode.done:
        print('Episode has been finished!!')
        policy_loss, value_loss = update_policy(agent, episode, discount_factor=DISCOUNT_FACTOR, \
                                                optimizer=optimizer, ppo_steps=PPO_STEPS, epsilon=EPSILON, \
                                                entropy_coefficient=ENTROPY_COEFFICIENT, batch_size=BATCH_SIZE, value_loss_coef=VALUE_LOSS_COEF)
        print(f'Policy Loss: {policy_loss}, Value Loss: {value_loss}')
        
        step_counter = 0
        episode_rewards.append(episode.rewards)
        episode_commands.append(command_list)
        command_list = []
        print(episode_commands)

        if episode_counter % 10 == 0:
            torch.save(agent.state_dict(), f'model_weights_{episode_counter}.pth')
            seriese = pd.DataFrame(episode_rewards)
            seriese.to_csv('./rewards.csv', encoding='cp949')
            commands = pd.DataFrame(episode_commands)
            commands.to_csv('./commands.csv', encoding='cp949')

        episode_counter += 1
        episode = None

        return jsonify({"status": "success", "control": "reset"})


    else:
        move_command, done = forward_pass(env, agent, episode=episode, now_state=now_state, episode_counter=episode_counter)
        move_command = action_to_direction[move_command]
        command_list.append(move_command)
        episode.state = now_state
        episode.done = done
        print(f'Step {step_counter} has been finished / state: {episode.state} / done: {episode.done}')
        if step_counter % 10 == 0 :
            print(command_list)
        step_counter += 1

    # if data.get("time", 0) > 15:
    # if step_counter > N_TRIALS:
    #    return jsonify({"status": "success", "control": "reset"})

    return jsonify({"status": "success", "message": "Data received"}), 200


@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    global now_state

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
    global move_command
    if move_command:
        command = move_command
        print(f'Sent Move Command: {command}')
        return jsonify({'move': command, 'weight': 0.8})
    else:
        return jsonify({"move": "STOP"})


@app.route('/get_action', methods=['GET'])
def get_action():
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
        global now_state
        now_state[2] = x_dest
        now_state[3] = z_dest
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
async def init():
    global rng
    random_coord = rng.integers(low=10, high=290, size=4)
    config = {
        "startMode": "pause",  # Options: "start" or "pause"
        "blStartX": int(random_coord[0]),  #Blue Start Position
        "blStartY": 10,
        "blStartZ": int(random_coord[1]),
        "rdStartX": int(random_coord[2]), #Red Start Position
        "rdStartY": 10,
        "rdStartZ": int(random_coord[3])
    }
    print("🛠️ Initialization config sent via /init:", config)
    send_reset_message()
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5253, debug=True)
