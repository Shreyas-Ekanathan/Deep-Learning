#implement an agent to play Kuhn poker against itself, CFR from scratch
import pyspiel
from collections import defaultdict
import random
import numpy as np

#based on https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf

#we are implementing tabular CFR, since this game is small enough
# will play with deep CFR later
# so the idea here is that we will keep a lookup table mapping curr state + history -> action
# and in training, we will have the model play against itself in every possible set of states (walk the full tree)
# then, it will calculate its regret for each of its actions and move to reduce that regret
# this is basically just dynamic programming

#key is state
# value is probability distribution over actions
regret_sum = defaultdict(lambda: defaultdict(float))
strategy_sum = defaultdict(lambda: defaultdict(float))

def CFR(state, prob0, prob1):
    #prob0 is the probability player 0 got to this state, likewise for p1
    if state.is_terminal():
        return np.array(state.returns()) #tells us teh result of the hand
    
    if state.is_chance_node(): #any fork in the road, a turn for the deck
        EV = 0.0 #how did we do in getting here?
        for outcome, p in state.chance_outcomes():
            EV += p * CFR(state.child(outcome), prob0, prob1) #accumulate EV
        return EV
    
    #now need to pick strategy for the player
    player = state.current_player()
    history = state.information_state_string(player)
    actions = state.legal_actions()
    strategy = regret_matching(history, actions) #dont need to deal with the action for training
    
    results = {}
    for action in actions:
        # above find out what happens if you take this action
        if player == 0:
            results[action] = CFR(state.child(action), prob0 * strategy[action], prob1)
        else:
            results[action] = CFR(state.child(action), prob0, prob1 * strategy[action])

    node_value = sum(strategy[a] * results[a] for a in actions)
    reach = prob0 if player == 0 else prob1 # my own reach
    cf_reach = prob1 if player == 0 else prob0  #other erach

    for action in actions:
        regret_sum[history][action] += cf_reach * (results[action][player] - node_value[player]) #immediate regret * p(they get here)
        strategy_sum[history][action] += reach * strategy[action] #build the average strategy: how often do i get here * result
        
    return node_value
        
def regret_matching(state, actions):
    regrets = regret_sum[state]
    regrets = {a: max(regrets[a], 0.0) for a in actions} 
            
    #now treat regrets as a probability distribution over the actions.
    #if regret is {0, 0.9, 2.1} -> play 2nd action 0.9/3, 3rd 2.1/3
    
    total = sum(regrets.values())
    
    if total > 0:
        return {a: regrets[a] / total for a in actions}
    else:
        return {a: 1.0 / len(actions) for a in actions} #just pick randomly
    
def pick_action(strategy):
    rand = random.random()
    total = 0
    for action, prob in strategy.items():
        total += prob
        if (rand < total):
            return action
    
def get_average_strategy():
    avg = {}
    for history, action_sums in strategy_sum.items():
        total = sum(action_sums.values())
        if total > 0:
            avg[history] = {a: s / total for a, s in action_sums.items()}
        else:
            n = len(action_sums)
            avg[history] = {a: 1.0 / n for a in action_sums}
    return avg

KUHN_ACTIONS = {0: "check/pass", 1: "bet/call"}
LEDUC_ACTIONS = {0: "fold", 1: "call/check", 2: "raise/bet"}

def print_strategy(strategy, action_names):
    for infoset in sorted(strategy):
        probs = strategy[infoset]
        parts = "  ".join(f"{action_names[a]} {p:5.1%}" for a, p in sorted(probs.items()))
        print(f"  {infoset:<12} {parts}")

#start on kuhn, check for nash equilibrium
game = pyspiel.load_game("kuhn_poker")

for i in range(10000):
    CFR(game.new_initial_state(), 1.0, 1.0)
    
strategy = get_average_strategy()
print("Kuhn Strategy!")
print_strategy(strategy, KUHN_ACTIONS)

# Kuhn Strategy!
#  0            check/pass 78.1%  bet/call 21.9%
#  0b           check/pass 100.0%  bet/call  0.0%
#  0p           check/pass 64.6%  bet/call 35.4%
#  0pb          check/pass 100.0%  bet/call  0.0%
#  1            check/pass 99.9%  bet/call  0.1%
#  1b           check/pass 67.0%  bet/call 33.0%
#  1p           check/pass 99.8%  bet/call  0.2%
#  1pb          check/pass 42.6%  bet/call 57.4%
#  2            check/pass 34.2%  bet/call 65.8%
#  2b           check/pass  0.0%  bet/call 100.0%
#  2p           check/pass  0.1%  bet/call 99.9%
#  2pb          check/pass  0.0%  bet/call 100.0%
#now for leduc
regret_sum.clear()
strategy_sum.clear()
game = pyspiel.load_game("leduc_poker")

for i in range(1000):
    CFR(game.new_initial_state(), 1.0, 1.0)
    
strategy = get_average_strategy()
print("Leduc Strategy!")
print_strategy(strategy, LEDUC_ACTIONS)

#now play leduc poker!

ACTION_NAMES = LEDUC_ACTIONS

play = 0

while play < 10:
    print("\nNew Hand!")
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            # deal a card 
            # sample it according to its probability.
            outcomes, probs = zip(*state.chance_outcomes())
            action = random.choices(outcomes, weights=probs)[0]
            state.apply_action(action)
            continue

        player = state.current_player()
        if (player == 0):
            print(f"Human to act")
            print(f"Info: {state.information_state_string(player)}")
            legal = state.legal_actions(player)
            options = ", ".join(f"{a}={ACTION_NAMES[a]}" for a in legal)
            while True:
                choice = input(f"Choose action [{options}]: ").strip()
                if choice.isdigit() and int(choice) in legal:
                    state.apply_action(int(choice))
                    break
                print("Invalid, try again")
        else:
            # computer
            history = state.information_state_string(player)
            action = pick_action(strategy[history])
            state.apply_action(action)

    print(f"\nCurrent State: {state}")
    print(f"Returns (payoffs): {state.returns()}")
    play += 1    
