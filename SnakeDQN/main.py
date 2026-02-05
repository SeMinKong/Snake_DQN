from game.snake_game import SnakeGameAI
from dqn.agent import Agent

def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    frame = game.reset()
    state_old = agent.get_state(frame)
    
    n_steps = 0
    print("Training Started (CNN Based)...")

    while True:
        final_move = agent.get_action(state_old)

        frame_new, reward, done, score = game.play_step(final_move, render=True)
        
        state_new = agent.get_state(frame_new)

        agent.remember(state_old, final_move, reward, state_new, done)

        n_steps += 1
        agent.train_replay()

        state_old = state_new

        if done:
            game.reset()
            agent.n_games += 1
            agent.update_epsilon()

            frame = game.get_frame()
            state_old = agent.get_state(frame)
            n_steps = 0

            if score > record:
                record = score
                agent.model.save('model_cnn_best.pth')

            print(f'Game {agent.n_games} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon:.3f}')

if __name__ == '__main__':
    train()