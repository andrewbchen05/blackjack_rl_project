from flask import Flask, request, jsonify, render_template
import rlcard
import torch
from rlcard.agents import DQNAgent as DQNAgent
from rlcard.agents import BlackjackHumanAgent as HumanAgent
from rlcard.utils.utils import print_card
import numpy as np

app = Flask(__name__)

# Initialize environment and agents
num_players = 2
env = rlcard.make('blackjack', config={'game_num_players': num_players})
human_agent = HumanAgent(env.num_actions)
dqn_agent = torch.load('rlcard/experiments/blackjack_dqn_result/model.pth')
env.set_agents([human_agent, dqn_agent])
chips_tally = [0] * num_players

def convert_to_serializable(data):
    """ Convert data to a serializable format. """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    else:
        return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def start_game():
    global chips_tally
    chips_tally = [0] * num_players
    trajectories, payoffs = env.run(is_training=False)
    
    # Convert data to serializable format
    trajectories_serializable = convert_to_serializable(trajectories)
    payoffs_serializable = convert_to_serializable(payoffs)
    
    return jsonify({'trajectories': trajectories_serializable, 'payoffs': payoffs_serializable})

@app.route('/play', methods=['POST'])
def play_game():
    action = request.json.get('action')
    # Process the action with the game environment
    # Update game state
    # Return updated game state and result
    return jsonify({'result': 'action processed'})

@app.route('/status', methods=['GET'])
def game_status():
    # Return current game status
    return jsonify({'status': 'game status'})

if __name__ == '__main__':
    app.run(debug=True)
