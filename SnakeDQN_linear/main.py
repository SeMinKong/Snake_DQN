from dqn.agent import Agent
from game.snake_game import SnakeGameAI

def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("Training Started (Vector Based with BFS)...")

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score {score} | Record {record}')

if __name__ == '__main__':
    train()