import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, output_size, input_channels=1):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)       
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)       
        
        self.fc_input_size = 64 * 7 * 7 
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, file_name='model_cnn.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, target_model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).to(self.device).float() / 255.0
        next_state = torch.from_numpy(next_state).to(self.device).float() / 255.0
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        pred = self.model(state)
        
        with torch.no_grad():
            next_pred = self.target_model(next_state)
        
        target = pred.clone()
        
        next_q_max = torch.max(next_pred, dim=1)[0]
        Q_new = reward + self.gamma * next_q_max * (~done)

        batch_indices = torch.arange(len(state), dtype=torch.long).to(self.device)
        action_indices = torch.argmax(action, dim=1)
        target[batch_indices, action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()