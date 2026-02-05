import torch
import random
import numpy as np
from collections import deque
from dqn.model import DQN, QTrainer

MAX_MEMORY = 100000 
BATCH_SIZE = 256
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        self.learn_step_counter = 0
        self.sync_target_frames = 1000

        self.model = DQN(output_size=3, input_channels=1).to(device)
        
        self.target_model = DQN(output_size=3, input_channels=1).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma, device=device)

    def get_state(self, frame):
        state = np.expand_dims(frame, axis=0)
        return state.astype(np.uint8)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
       
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.sync_target_frames == 0:
            self.update_target_network()
            print("--- Target Network Updated ---")

    def get_action(self, state):
        final_move = [0, 0, 0]
        
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0) / 255.0
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())