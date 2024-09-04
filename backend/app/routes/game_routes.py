from flask import Blueprint, request, jsonify
import torch
import rlcard

# Add the rlcard directory to sys.path


# Import agents from the rlcard module
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents import BlackjackHumanAgent as HumanAgent
from rlcard.utils.utils import print_card

game_routes = Blueprint('game_routes', __name__)

# Load your trained model
dqn_agent = torch.load('rlcard/experiments/blackjack_dqn_result/model.pth')

@game_routes.route('/play', methods=['POST'])
def play_blackjack():
    try:
        # Get the action from the request
        user_action = request.json.get('action', 'stand')  # Default to 'stand' if no action provided

        num_players = 2
        env = rlcard.make(
            'blackjack',
            config={
                'game_num_players': num_players,
            }
        )
        
        # Initialize the agents
        human_agent = HumanAgent(env.num_actions)
        env.set_agents([human_agent, dqn_agent])

        # Run the game
        trajectories, payoffs = env.run(is_training=False)

        # Process and format game results
        result = process_game_results(trajectories, payoffs)
        
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def process_game_results(trajectories, payoffs):
    final_state = []
    action_record = []
    state = []
    _action_list = []

    for i in range(len(trajectories)):
        final_state.append(trajectories[i][-1])
        state.append(final_state[i]['raw_obs'])

    action_record.append(final_state[i]['action_record'])
    for i in range(1, len(action_record) + 1):
        _action_list.insert(0, action_record[-i])

    game_details = {
        'actions': [],
        'dealer_hand': state[0]['state'][1],
        'players_hands': [state[i]['state'][0] for i in range(1, len(state))],
        'results': [{'player': i, 'payoff': payoffs[i]} for i in range(len(payoffs))]
    }

    for pair in _action_list[0]:
        game_details['actions'].append({'player': pair[0], 'action': pair[1]})

    return game_details
