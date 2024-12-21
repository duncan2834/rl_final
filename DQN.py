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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
epsilon_decay = 0.998
epsilon_min = 0.01
learning_rate = 0.001
gamma = 0.99
batch_size = 256
num_episodes = 50
buffer_capacity = 80000
target_update_freq = 10

# Hàm training
def train_q_network(q_network, target_network, buffer, optimizer, batch_size, gamma):
    # khi chưa lấy đủ mẫu
    if len(buffer) < batch_size:
        return
    
    # Lấy ngẫu nhiên 1 batch từ replay buffer
    state, action, reward, next_state, done = buffer.sample(batch_size)
    
    # Tính q_value
    q_values = q_network(state) # [batchsize, 21]
    q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # 
    
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
env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
    dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=3,
    max_cycles=1000, extra_features=False)

obs_shape = env.observation_space("blue_0").shape # [13, 13, 5]
action_shape = env.action_space("blue_0").n

replay_buffer = ReplayBuffer(buffer_capacity)
q_networks_dict = {}
target_networks_dict = {}
optimizers_dict = {}

for count in range(9): # chạy qua các agent và khởi tạo các q network, target network, optimizer cho các agent blue
     # chia thành 9 nhóm, mỗi nhóm 9 agent blue. 0 -> 8
    q_networks_dict[count] = DQN(obs_shape, action_shape)
    target_networks_dict[count] = DQN(obs_shape, action_shape)
    target_networks_dict[count].load_state_dict(q_networks_dict[count].state_dict())
    optimizers_dict[count] = optim.Adam(q_networks_dict[count].parameters(), lr=learning_rate)

# Training loop
for ep in range(num_episodes):
    env.reset()
    total_reward = 0
    count_step = 0
    for agent in env.agent_iter(): # chạy qua tất cả các agent (red 0 blue 0 ,....)
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None # agent đi rồi
        else:
            agent_handle = agent.split("_")[0] # ví dụ là red_0 thì lấy red 
            agent_index = int(agent.split("_")[1]) 
            if agent_handle == "red": 
                action = env.action_space(agent).sample()
            else: # agent blue
                state_tensor = (
                    torch.tensor(observation, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)   
                ) # lúc đầu là [height, weight, channels] đổi sang [channels, height, weight] xong unsqueeze(0) để thêm 1 chiều batch_size = 1
                # mấy agent 0 -> 8 thì thuộc mạng 0, 9 -> 17 là mạng 1, .... 80 thuộc mạng 8
                    
                q_network = q_networks_dict[count_step]
                target_network = target_networks_dict[count_step]
                optimizer = optimizers_dict[count_step]
                
                if (agent_index + 1) % 9 == 0:
                    count_step += 1
                if count_step == 9:
                    count_step = 0
                # select action theo epsilon-greedy
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        q_values = q_network(state_tensor)  # forward
                        action = q_values.argmax().item() # lấy chỉ số có q to nhất

        obs = observation # trạng thái trước khi hành động
        env.step(action)
        if agent_handle == "blue":
            if not termination and not truncation: # agent chưa chết thì add vào replaybuffer
                observation, reward, termination, truncation, info = env.last()
                replay_buffer.add(obs, action, reward, observation, termination or truncation)
            # state và next_state [batchsize, channels, height, weight]
            total_reward += reward
    # Train DQN tren blue agent
    for i in range(9):
        train_q_network(q_networks_dict[i], target_networks_dict[i], replay_buffer, optimizers_dict[i], batch_size, gamma)
        
    # Update target network
    if ep % target_update_freq == 0:
        for key in target_networks_dict: # duyet qua dict, key la agent_index
            target_networks_dict[key].load_state_dict(q_networks_dict[key].state_dict())
        
    if epsilon > epsilon_min: 
        epsilon = epsilon * epsilon_decay
    print(epsilon)
    print(f"Episode {ep + 1}: Total Reward: {total_reward}")
    
torch.save(q_network.state_dict(), "blue_final.pt")
env.close()
    