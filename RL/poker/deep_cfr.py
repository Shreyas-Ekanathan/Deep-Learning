# implement deep cfr to play poker 
# partially based on https://arxiv.org/pdf/1811.00164

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pyspiel
import numpy as np
import random
import os

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        #future update: could embed card positions
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape)
        )
    
    def forward(self, input):
        return self.layers(input)
        
def deep_CFR(state, player, regret_buffer, strategy_buffer, t):
    #prob0 is the probability player 0 got to this state, likewise for p1
    if state.is_terminal():
        return state.returns()[player] #tells us teh result of the hand
    
    if state.is_chance_node(): #any fork in the road, a turn for the deck
        outs, ps = zip(*state.chance_outcomes())
        return deep_CFR(state.child(random.choices(outs, weights=ps)[0]), player, regret_buffer, strategy_buffer, t)    
    
    #now need to pick strategy for the player
    curr_player = state.current_player()
    input = state.information_state_tensor(curr_player) 
    actions = state.legal_actions()

    with torch.no_grad():
        strat = regret_matching(regret_predictors[curr_player](torch.tensor(input, dtype=torch.float32)), actions)
        
    if (player != curr_player):
        opt = random.choices(list(strat), weights=list(strat.values()))[0] #just do something
        strat_vector = np.zeros(game.num_distinct_actions(), dtype=np.float32)
        for a in actions:
            strat_vector[a] = strat[a]
        add(strategy_buffer[player], ((np.array(input), t, strat_vector))) #store for training
        return deep_CFR(state.child(opt), player, regret_buffer, strategy_buffer, t)

    #our turn, lets do something!
    values, node_value = {}, 0.0
    for a in actions: #sample each aciton
        values[a] = deep_CFR(state.child(a), player, regret_buffer, strategy_buffer, t)
        node_value += strat[a] * values[a] #EV

    regret = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    for a in actions:
        regret[a] = values[a] - node_value #could we have done better than average?
        
    add(regret_buffer[player], ((np.array(state.information_state_tensor(player)), t, regret)))
    return node_value

def regret_matching(regrets, actions):
    regrets = {a: max(regrets[a].item(), 0.0) for a in actions}
            
    #now treat regrets as a probability distribution over the actions.
    #if regret is {0, 0.9, 2.1} -> play 2nd action 0.9/3, 3rd 2.1/3
    
    total = sum(regrets.values())
    
    if total > 0:
        return {a: regrets[a] / total for a in actions}
    else:
        return {a: 1.0 / len(actions) for a in actions} #just pick randomly

game = pyspiel.load_game("universal_poker", {
    "betting": "limit", "numPlayers": 2, "numRounds": 2,
    "numSuits": 4, "numRanks": 13, "numHoleCards": 2, "numBoardCards": "0 3",
    "blind": "1 1", "raiseSize": "2 4", "maxRaises": "2 2", "firstPlayer": "1 1",
})

input_shape = game.information_state_tensor_size()
regret_predictors = [MLP(input_shape, 3) for i in range(2)]
strategy = MLP(input_shape, 3)

regret_buffer = [[], []]
strategy_buffer = [[], []]
buffer_limit = 1000000
def add(buf, item):
    if len(buf) < buffer_limit:
        buf.append(item)
    else:
        buf[random.randrange(buffer_limit)] = item # random replacement once full

num_epochs = 50
num_samples = 10000
root = game.new_initial_state()

class CFRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        info, t, target = self.samples[i]
        return (torch.tensor(info, dtype=torch.float32), torch.tensor(float(t), dtype=torch.float32), torch.tensor(target, dtype=torch.float32))

for epoch in range(num_epochs):
    average_loss = 0
    n = 0
    for player in [0, 1]:
        for k in range(num_samples):
            deep_CFR(root, player, regret_buffer, strategy_buffer, epoch + 1) #monte carlo sampling (per MCCFR)
        #reinit 
        loader = DataLoader(CFRDataset(regret_buffer[player]), batch_size=128, shuffle=True)
        regret_predictors[player] = MLP(input_shape, 3)
        optimizer = torch.optim.Adam(regret_predictors[player].parameters(), lr=1e-3)
        #train
        for (iteration, (inputs, t, target)) in enumerate(loader):
            optimizer.zero_grad()
            predicted_regrets = regret_predictors[player](inputs)
            per_sample = ((predicted_regrets - target) ** 2).sum(dim=1)  
            loss = (t * per_sample).sum() / t.sum() #weighted by epoch
            average_loss += loss.item()
            n += 1
            loss.backward()
            optimizer.step()
            if (iteration > 5000): break #cap training
            
    average_loss = average_loss / n
    print(f"Epoch: {epoch + 1}, Average Loss = {average_loss}")

#train strategy_net on strategy buffer
strategy_inputs = strategy_buffer[0] + strategy_buffer[1]
loader = DataLoader(CFRDataset(strategy_inputs), batch_size=128, shuffle=True)
optimizer = torch.optim.Adam(strategy.parameters(), lr=1e-3)
num_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
for epoch in range(num_epochs):
    average_loss = 0
    for info, t, target in loader:
        optimizer.zero_grad()
        logits = strategy(info)
        probs = F.softmax(logits, dim=1) #softmax to get p distr over actions
        per_sample = ((probs - target) ** 2).sum(dim=1)
        loss = (t * per_sample).sum() / t.sum() #weight by epoch
        average_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    average_loss = average_loss / len(loader)
    print(f"Epoch: {epoch + 1}, Average Loss = {average_loss}")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_net.pth")
torch.save(strategy.state_dict(), save_path)

#evals
POKER_ACTIONS = {0: "fold", 1: "call/check", 2: "raise/bet"}
@torch.no_grad()
def eval_strategy(net, num_hands=500, max_print=80):
    seen = {}

    def strat_at(state):
        cur = state.current_player()
        legal = state.legal_actions()
        key = state.information_state_string(cur)
        if key not in seen: # compute the net once 
            info = torch.tensor(state.information_state_tensor(cur), dtype=torch.float32)
            logits = net(info)
            masked = torch.full((game.num_distinct_actions(),), float("-inf"))
            for a in legal:
                masked[a] = logits[a]
            probs = torch.softmax(masked, dim=0)   
            seen[key] = {a: probs[a].item() for a in legal}
        return seen[key]

    for i in range(num_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outs, ps = zip(*state.chance_outcomes())
                state.apply_action(random.choices(outs, weights=ps)[0])
            else:
                dist = strat_at(state)
                state.apply_action(random.choices(list(dist), weights=list(dist.values()))[0])

    print(f"\nLearned strategy (deep CFR): {len(seen)} infosets over {num_hands} sampled hands:")
    for key in sorted(seen)[:max_print]:
        dist = "  ".join(f"{POKER_ACTIONS[a]} {p:5.1%}" for a, p in sorted(seen[key].items()))
        print(f"  {key}\n      {dist}")

eval_strategy(strategy)

#play against the bot
def bot_action(net, state):
    cur = state.current_player()
    legal = state.legal_actions()
    info = torch.tensor(state.information_state_tensor(cur), dtype=torch.float32)
    masked = torch.full((game.num_distinct_actions(),), float("-inf"))
    logits = net(info)
    for a in legal:
        masked[a] = logits[a]
    probs = torch.softmax(masked, dim=0) # distribution over legal actions
    return random.choices(legal, weights=[probs[a].item() for a in legal])[0]

def play_vs_bot(net, num_hands=10):
    for i in range(num_hands):
        print("\nNew Hand!")
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node(): # dec
                outs, ps = zip(*state.chance_outcomes())
                state.apply_action(random.choices(outs, weights=ps)[0])
                continue
            player = state.current_player()
            if player == 0: # you
                print("Your turn.")
                print(f"  Info: {state.information_state_string(player)}")
                legal = state.legal_actions(player)
                options = ", ".join(f"{a}={POKER_ACTIONS[a]}" for a in legal)
                while True:
                    choice = input(f"  Choose action [{options}]: ").strip()
                    if choice.isdigit() and int(choice) in legal:
                        state.apply_action(int(choice))
                        break
                    print("  invalid, try again")
            else: # trained bot
                a = bot_action(net, state)
                print(f"Bot plays: {POKER_ACTIONS[a]}")
                state.apply_action(a)
        print(f"Result: {state}")
        print(f"Returns (payoffs): {state.returns()}   (you are player 0)")

play_vs_bot(strategy)
