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
import ppo
import torch

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
import tank_env
import torch.optim as optim

gym.register(
    id="gymnasium_env/TankEnv-v0",
    entry_point=tank_env.TankEnv,
)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    rad = np.deg2rad(heading)
    sin = np.sin(rad)
    cos = np.cos(rad)
    return sin, cos

# Initialize Flask server
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# log.disabled = False

# Load YOLO model
model = YOLO('yolov8n.pt')

# Action commands
action_command = ["FIRE"]

MAX_EPISODES = 300              
DISCOUNT_FACTOR = 0.99
REWARD_THRESHOLD = 12
PRINT_INTERVAL = 10
PPO_STEPS = 10
N_TRIALS = 512
EPSILON = 0.2
ENTROPY_COEFFICIENT = 0.1
HIDDEN_DIMENSIONS = 64
DROPOUT = 0.2
LEARNING_RATE = 0.0002
BATCH_SIZE = 128
VALUE_LOSS_COEF = 0.5
RANDOM_EPISODE = 30

env = gym.make('gymnasium_env/TankEnv-v0', max_steps=N_TRIALS, threshold = REWARD_THRESHOLD)
rng = np.random.default_rng(seed=13)

# 모델 초기화 // 모델 웨이트 입혀서 수행 가능.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ppo.create_agent(env, hidden_dimensions=HIDDEN_DIMENSIONS, dropout=DROPOUT)
# checkpoint = torch.load("model_weights_200.pth", map_location=device)
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
        policy_loss, value_loss = ppo.update_policy(agent, episode, discount_factor=DISCOUNT_FACTOR, \
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
        move_command, done, info = ppo.forward_pass(env, agent, episode=episode, grid_epsilon = ENTROPY_COEFFICIENT * (1 - episode_counter/MAX_EPISODES), episode_count = episode_counter, random_episode=RANDOM_EPISODE)
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
    curriculum = episode_counter + REWARD_THRESHOLD + 10
    while True:
        random_coord = rng.integers(low=10, high=290, size=4)
        x = int(random_coord[0])
        z = int(random_coord[1])
        if episode_counter < 120:
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
    app.run(host='0.0.0.0', port=5257, debug=True)
