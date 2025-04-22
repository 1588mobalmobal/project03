import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions
import numpy as np
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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