import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from magent2.environments import battle_v4

# observation_shape [13, 13, 5] [height, weight, channels]
# action shape [21]
# định dạng đầu vào của Conv2d là [batchsize, channels, height, weight]
class DQN(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        # CNN, kernel 3x3
        
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)    
        flatten_dim = dummy_output.view(-1).shape[0]  # flatten thành vector 1D
        # 3 lớp Fully Connected, sử dụng ước tính giá trị Q cho các action.
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )
        
    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        # kiểu x có dạng [batch_size, channels, height, width]
        x = self.cnn(x) # đi qua CNN để trích xuất features
        if len(x.shape) == 3: # khi không dùng batch
            batchsize = 1
        else:
            batchsize = x.shape[0] # khi dùng batch thì lấy giá trị đầu là batch_size
        x = x.reshape(batchsize, -1) # chuyển thành vector 1D
        return self.network(x)
    
# Replay Buffer lưu trải nghiệm
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # danh sách 2 đầu, đạ maxlen thì loại bỏ các phần tử cũ nhất khi thêm phần tử mới

    # thêm trải nghiệm vào buffer
    # state 
    # action
    # reward
    # next_state
    # termination
    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1) # [batch_size, channels, height, width]
        next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1)  # [batch_size, channels, height, width]
        self.buffer.append((state, action, reward, next_state, done))

    # lấy ngẫu nhiên 1 batch ứng với batch size
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32), # kết hợp tất cả các trạng thái trong batch thành một tensor
            torch.tensor(np.array(action), dtype=torch.long),
            torch.tensor(np.array(reward), dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )

    # trả lại độ dài buffer
    def __len__(self):
        return len(self.buffer)


# Hyperparameters
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
gamma = 0.99
batch_size = 64
num_episodes = 10
buffer_capacity = 10000
target_update_freq = 1000

# Hàm training
def train_q_network(env, q_network, target_network, buffer, optimizer, batch_size, gamma):
    # khi chưa lấy đủ mẫu
    if len(buffer) < batch_size:
        return
    
    # Lấy ngẫu nhiên 1 batch từ replay buffer
    state, action, reward, next_state, done = buffer.sample(batch_size)
    
    # Tính q_value
    q_values = q_network(state)
    q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    
    # Tính q_value target
    with torch.no_grad():
        next_q_value = target_network(next_state).max(1)[0]
        target_q_values = reward + gamma * next_q_value * (1 - done)
        
    # Tính loss
    loss = nn.MSELoss()(q_values, target_q_values)
    
    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# Environment and setup
env = battle_v4.env(map_size=45)
obs_shape = env.observation_space("blue_0").shape # [13, 13, 5]
action_shape = env.action_space("blue_0").n


q_network = DQN(obs_shape, action_shape) # init
target_network = DQN(obs_shape, action_shape) # init
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_capacity)

# Training loop
for ep in range(num_episodes):
    env.reset()
    total_reward = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            state_tensor = (
                torch.tensor(observation, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
            ) # lúc đầu là [height, weight, channels] đổi sang [channels, height, weight] xong unsqueeze(0) để thêm 1 chiều batch_size = 1
            # select action theo epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state_tensor)  # forward
                    action = q_values.argmax().item()
                    
        state = observation # trạng thái trước khi hành động
        env.step(action)

        if not termination and not truncation: # nếu chưa hết thì add vào replay buffer
            replay_buffer.add(state, action, reward, observation, termination or truncation)
            # state và next_state [batchsize, channels, height, weight]
        
        total_reward += reward
        
        # Train DQN
        train_q_network(env, q_network, target_network, replay_buffer, optimizer, batch_size, gamma)
        
    # Update target network
    if ep % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
        
    if epsilon > epsilon_min: 
        epsilon = epsilon * epsilon_decay
    print(f"Episode {ep + 1}: Total Reward: {total_reward}")
    
torch.save(q_network.state_dict(), "blue.pt")
env.close()
    
            
            
            
