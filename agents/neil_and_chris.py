"""
Agent for Kaggle's Candy Cane XMas game, 2020

This agent combines two variants of a UCB-like strategy.

The estimated reward is based on a Bayesian distribution model of own pull data and a model of decay,
augmented by a neural network model that tries to predict the true expected reward, using knowledge of
opponent behaviour.

To this is added a UCB-like exploration bonus, based on uncertainty in the original Bayesian estimates. The
bonus is multiplied by a factor that varies over time, starting high to encourage exploration, and reducing
to low values towards the end of the game.

Finally, and where the variants most differ, an extra "follow the opponent" bonus is added based on a trace
of opponent's behaviour.

The selected action is always the argmax of this score function calculated for all bandits.

The decision whether to take agentA's or agentB's suggested action is managed by a classifier trained to 
select the most effective strategy depending on statistics about the opponent. This meta-policy could be
extended into a reinforcement-learning approach with more options (and less tuning of general parameters),
and that might be tuned through self-play, but we ran out of time to try out that idea.
"""

import numpy as np
import math
import torch
from torch import nn
from torch import tensor
import base64
import io

# Params for estimator
REP_TRACE_FORGET = 0.9             # Decay to forget the opponent's replacing trace in general
REP_TRACE_ERASE = 0.7              # Additional decay to opponent replacing trace due to my choice
ACC_TRACE_FORGET = 0.975           # Decay to forget the opponent's accumulating trace in general
ACC_TRACE_ERASE = 0.2              # Additional decay to opponent accumulating trace due to my choice

# Params for action selector (agentA)
AGENTA_START_C = 0.4               # UCB multiplier at t=0
AGENTA_MIDPOINT = 900              # UCB midpoint
AGENTA_MID_C = 0.0                 # UCB multiplier at t=AGENTA_MIDPOINT
AGENTA_END_C = 0.1                 # UCB multiplier at t=2000
AGENTA_TRACE_SELECT_FACTOR = 0.05  # Mulitplier for trace add
AGENTA_TSX = 1.75                  # Schedule factor for t=0..600 for TSA
AGENTA_TSY = 1.0                   # Schedule factor for t=600..1200 for TSA
AGENTA_TSZ = 0.4                   # Schedule factor for t=1200.. for TSA

# Params for action selector (agentB)
AGENTB_C = 0.35                    # UCB multiplier
AGENTB_C_OFFSET = 0.05             # C scheduling offset
AGENTB_TRACE_SELECT_FACTOR = 0.15  # Mulitplier for trace add

# Globals
total_reward = 0
estimator = None
selector = None
predictor = None
nn_model_rewards = None
nn_model_oppclass = None


###################################################################################################


class RewardPredictorNN(nn.Module):
    """
    Neural network model. Input is features of an individual bandit, including Bayesian expected
    from own pulls, and stats about opponent. Output is expected reward from that bandit.
    """

    def __init__(self, input_size=11, hsize=32):
        super(RewardPredictorNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hsize),
            nn.Tanh(),
            nn.Linear(hsize, hsize),
            nn.Tanh(),
            nn.Linear(hsize, hsize),
            nn.Tanh(),
            nn.Linear(hsize, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


nn_model_rewards = RewardPredictorNN()


###################################################################################################


class OpponentClassifierNN(nn.Module):
    """
    Neural network model. Input is aggregate features of game so far. 
    Output is prediction of whether the opponent is vulnerable to agentB.
    """

    def __init__(self, input_size=7, hsize=16):
        super(OpponentClassifierNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hsize),
            nn.Tanh(),
            nn.Linear(hsize, hsize),
            nn.Tanh(),
            nn.Linear(hsize, hsize),
            nn.Tanh(),
            nn.Linear(hsize, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


nn_model_oppclass = OpponentClassifierNN()


##################################################################################################


class RewardPredictor():
    """
    Tracks features for using neural network to predict expected reward, and the current set of
    predictions. Runs the neural network as part of its update step.
    """

    def __init__(self, model, n_bandits=100):
        self.ts = 0                                        # Current game timestamp
        self.nn = model                                    # The neural network model
        self.expected_via_model = np.full(n_bandits, 0.5)  # Estimates refined using predictive model over features

        # For normalisation, data source is vga0125
        self.features_mean = np.array([1053.8690208204387, 0.27609418796223606, 0.14942522307672723, 0.37882489454723095, 10.593304452145176, 10.5944149667108, 870.4149880189414, 0.3411261426628257, 0.333614352676285, 0.7379720207826987, 2.1987828138558987], dtype=np.float32)
        self.features_std =  np.array([548.265463854059, 0.1184266620971515, 0.09233582986852824, 0.1849428151854555, 10.048786059373061, 9.386191815113945, 590.4904244556535, 0.30619695028683586, 0.14004013353721168, 0.11745975792825801, 0.6505550181732546], dtype=np.float32)

    def init_features(self, belief_state, action_track):
        """
        Set up default features for neural network input.
        """

        n_bandits = belief_state.n_bandits
        all_features = np.full((n_bandits, 11), 0.0, dtype=np.float32)
        indiv_feature = self.calc_features(belief_state, action_track, 0)
        all_features += indiv_feature
        self.all_features = all_features


    def update_global_features(self, belief_state, action_track):
        """
        Bulk update to some input features for all bandits, where features relate to general game state.
        """

        self.all_features[:, 0] = (self.ts - self.features_mean[0])/self.features_std[0]
        self.all_features[:, 8] = (action_track.opp_bunching.mean() - self.features_mean[8])/self.features_std[8]
        self.all_features[:, 9] = (action_track.calc_overlap() - self.features_mean[9])/self.features_std[9]
        self.all_features[:, 10] = (action_track.calc_opp_pull_std() - self.features_mean[10])/self.features_std[10]


    def update_action_features(self, belief_state, action_track, action):
        """
        Update to a single bandit's feature, used when self or opponent pull it.
        """

        self.all_features[action] = self.calc_features(belief_state, action_track, action)


    def calc_features(self, belief_state, action_track, action):
        """
        Neural network features for an individual action
        """

        features = np.array([
            [self.ts, belief_state.expected[action], belief_state.est_lower_quantile[action], belief_state.est_upper_quantile[action],
             action_track.n_my_selections[action],  action_track.n_opp_selections[action], action_track.opp_last_ts[action], action_track.opp_bunching[action],
             action_track.opp_bunching.mean(), action_track.calc_overlap(), action_track.calc_opp_pull_std()]
        ], dtype=np.float32)

        features = (features - self.features_mean)/self.features_std

        return features


    def run_model_update_batch(self):
        """
        Run the neural network forward for all current features and update expected values
        """

        minibatch = tensor(self.all_features)
        self.expected_via_model[:] = self.nn(minibatch).detach().numpy().flatten()


    def update_step(self, belief_state, action_track, ts, action, opp_action):
        """
        Main entry point for updating NN features.
        """

        self.ts = ts
        self.update_action_features(belief_state, action_track, action)
        self.update_action_features(belief_state, action_track, opp_action)
        self.update_global_features(belief_state, action_track)
        self.run_model_update_batch()


####################################################################################################


class BeliefState():
    """
    BeliefState tracks most likely distribution of bandit ratings and decay rates based on knowledge
    of actions taken and reward gained only. It does not attempt to track effects of opponent moves other
    than known decay factor.

    Updates are based on Bayes rule.
    """

    def __init__(self, n_bandits=100, n_bins=101, decay=0.97, lcent_p=0.2, ucent_p=0.8):
        self.ts = 0
        self.n_bandits = n_bandits         # Number of bandits in environment
        self.n_bins = n_bins               # Number of possible ratings
        self.decay = decay                 # Decay rate after each pull

        self.lower_quantile_p = lcent_p     # Lower quantile cumulative probability point
        self.upper_quantile_p = ucent_p     # Upper quantile cumulative probability point

        self.ratings = np.linspace(0.0, 100.0, n_bins)      # Value of each rating (actually just [0..100] for 101 bins)

        self.dist = np.full((n_bandits, n_bins), 1.0/n_bins )  # "Belief state" of orginal bandit ratings
        self.cprobs = np.cumsum(self.dist, axis=1)          # Cumulative probabilities from "belief state"
        self.decay_factor = np.full(n_bandits, 1.0)         # Current decay factor on each bandit

        self.expected = np.full(n_bandits, 0.5)                      # Estimates for expected reward value on next pull, mean over belief distribution
        self.est_lower_quantile = np.full(n_bandits, lcent_p)         # Estimates for reward value on next pull at lower quantile
        self.est_upper_quantile = np.full(n_bandits, ucent_p)         # Estimates for reward value on next pull at upper quantile
        self.quantile_width = self.est_upper_quantile - self.est_lower_quantile  # Difference in expected reward between upper and lower quantile


    def calc_expected(self, action):
        """
        Expected reward, based on Bayes updates from observations of own pulls plus total number of pulls.
        """

        return (self.dist[action] * np.ceil(self.ratings * self.decay_factor[action])).sum()/101.0


    def bayes_update(self, action, result):
        """
        For own pulls only, maintains a belief state over the distribution of original bandit ratings.
        """

        priors = self.dist[action, :]
        p_result_given_ratings = np.ceil(self.ratings * self.decay_factor[action])/101.0
        if not result:
            p_result_given_ratings = 1.0 - p_result_given_ratings
        # Apply Bayes, ignore denominator for p(result) because it is same for all guesses and we are normalising at the end
        self.dist[action, :] = p_result_given_ratings * priors
        self.dist[action, :] = self.dist[action, :]/self.dist[action, :].sum()


    def update_cprobs(self, action):
        """
        Maintains cumulative distribution probabilies for single action.
        """

        self.cprobs[action] = np.cumsum(self.dist[action])


    def calc_exp_return_at_quantile(self, action, cent_p):
        """
        Calculates quantile point expectation for given action, based on quantile_p being tracked.
        """

        idx = int(np.argmax(self.cprobs[action] > cent_p))
        if idx == 0:
            return 0.0

        upper_p = self.cprobs[action, idx]
        lower_p = self.cprobs[action, idx-1]
        ratio = (cent_p - lower_p)/(upper_p - lower_p)
        low_return = np.ceil( (idx-1) * self.decay_factor[action] )/self.n_bins
        high_return = np.ceil( idx * self.decay_factor[action] )/self.n_bins
        return ratio * high_return + (1-ratio) * low_return


    def arbitary_quantile(self, c=0.5):
        """
        Calculates quantile point expectation over all actions.
        """

        upper_idx = np.argmax(self.cprobs > c, axis=1)
        lower_idx = np.clip(upper_idx - 1, 0, 99)

        upper_p = self.cprobs[np.arange(self.n_bandits), upper_idx]
        lower_p = self.cprobs[np.arange(self.n_bandits), lower_idx]

        ratio = (c - lower_p)/(upper_p - lower_p)
        low_return = np.ceil( lower_idx * self.decay_factor )/self.n_bins
        high_return = np.ceil( upper_idx * self.decay_factor )/self.n_bins
        return ratio * high_return + (1-ratio) * low_return


    def decay_update(self, action):
        """
        Tracks effect of decay on each bandit. This is tracked separately to the discrete belief state.
        """

        self.decay_factor[action] *= self.decay


    def update_belief_stats(self, action):
        """
        Updates the expected return, plus upper and lower quantile expected returns from belief state and known decays.

        This ignores guesses due to opponent's unknown results.
        """

        self.expected[action] = self.calc_expected(action)

        self.est_lower_quantile[action] = self.calc_exp_return_at_quantile(action, self.lower_quantile_p)
        self.est_upper_quantile[action] = self.calc_exp_return_at_quantile(action, self.upper_quantile_p)
        self.quantile_width[action] = self.est_upper_quantile[action] - self.est_lower_quantile[action]


    def update_step(self, ts, action, result, opp_action):
        """
        Main entry point for updating belief state object. It updates all properties in correct order,
        given observation data from a single step.
        """

        self.ts = ts

        self.bayes_update(action, result)
        self.update_cprobs(action)

        self.decay_update(action)
        self.decay_update(opp_action)

        self.update_belief_stats(action)
        self.update_belief_stats(opp_action)


####################################################################################################


class ActionTracker():
    """
    ActionTracker collects stats on own and opponent's choices.
    """

    def __init__(self, n_bandits=100, rep_trace_forget=0.8, rep_trace_erase=1.0, acc_trace_forget=0.95, acc_trace_erase=0.3):
        self.ts = 0
        self.n_bandits = n_bandits         # Number of bandits in environment

        self.opp_rep_trace = np.full(n_bandits, 0.0)      # Replacinging trace of opponent pulls for guided exploration
        self.opp_acc_trace = np.full(n_bandits, 0.0)      # Accumulating trace of opponent pulls for guided exploration

        self.trace_ready = False                          # True when something is in the trace
        self.rep_trace_forget = rep_trace_forget          # Forget rate for opponent pulls trace (replacing)
        self.rep_trace_erase = rep_trace_erase            # Erase factor for opponent trace when visited (replacing)
        self.acc_trace_forget = acc_trace_forget          # Forget rate for opponent pulls trace (accumulating)
        self.acc_trace_erase = acc_trace_erase            # Erase factor for opponent trace when visited (accumulating)

        self.n_my_selections = np.full(n_bandits, 0.0)    # Number of own pulls for each bandit
        self.n_opp_selections = np.full(n_bandits, 0.0)   # Number of opponent pulls for each bandit

        self.opp_last_ts = np.full(n_bandits, 0.0)         # Last time step each bandit pulled by opponent
        self.opp_bunching = np.full(n_bandits, 0.0)        # Metric that is higher when opponent pulls same bandit more frequently in bursts

        # These track how the opponent reacts to our choices
        self.my_trace_a = np.full(n_bandits, 0.0)          
        self.opponent_mean_trace_a = 0.0
        self.my_trace_b = np.full(n_bandits, 0.0)
        self.opponent_mean_trace_b = 0.0


    def update_bunching_stats(self, opp_action):
        """
        Updates a bunching metric for opponent's pulls.

        The metric is higher when opponent repeats same action with fewer timesteps between pulls.
        """

        ts = self.ts
        n_opp = self.n_opp_selections[opp_action]
        if n_opp > 1:
            this_bunch = 1.0/np.sqrt(ts - self.opp_last_ts[opp_action])
            self.opp_bunching[opp_action] += (this_bunch - self.opp_bunching[opp_action])/(n_opp-1)

        self.opp_last_ts[opp_action] = ts

    def update_opponent_trace(self, opp_action, action):
        """
        Updates a recency trace for opponent's pulls.
        """

        self.opp_rep_trace[:] *= self.rep_trace_forget
        self.opp_rep_trace[action] *= self.rep_trace_erase

        self.opp_acc_trace[:] *= self.acc_trace_forget
        self.opp_acc_trace[action] *= self.acc_trace_erase

        if self.n_opp_selections[opp_action] < 2:
            return

        self.trace_ready = True
        self.opp_rep_trace[opp_action] = 1
        self.opp_acc_trace[opp_action] += 1

    def update_classifier_trace(self, opp_action, action):
        """
        Updates reverse traces to classify opponent style. Trace forget rates are
        fixed, to match data capture routines used in training data.
        """

        ts = self.ts
        self.opponent_mean_trace_a += (1.0/ts) * (self.my_trace_a[opp_action] - self.opponent_mean_trace_a)
        self.my_trace_a *= 0.95
        self.my_trace_a[action] += 1

        self.opponent_mean_trace_b += (1.0/ts) * (self.my_trace_b[opp_action] - self.opponent_mean_trace_b)
        self.my_trace_b *= 0.9
        self.my_trace_b[action] = 1

    def update_step(self, ts, action, opp_action):
        """
        Main entry point for updating statistics object. It updates all properties in correct order,
        given observation data from a single step.
        """

        self.ts = ts
        self.n_my_selections[action] += 1
        self.n_opp_selections[opp_action] += 1

        self.update_bunching_stats(opp_action)
        self.update_opponent_trace(opp_action, action)
        self.update_classifier_trace(opp_action, action)


    def calc_overlap(self):
        """
        An overlap metric, that measures how much own and opponents choices coincide across all bandits.

        If own and opponent choices are identical, the metric scores 1.0, if they are completely separate,
        the metric scores close to 0.0
        """

        anti_overlap = np.abs(self.n_my_selections - self.n_opp_selections).sum()
        return (1 - anti_overlap/(2*self.ts+2))


    def calc_opp_pull_std(self):
        """
        The standard deviation of opponent pull numbers, normalised so that the expected std for a random agent
        is 1.0.
        """

        if self.ts < 1.0:
            return 1.0
        return self.n_opp_selections.std()/np.sqrt(self.ts/101.010101)


###################################################################################################


class CombinedEstimator():
    """
    Collection of a number of possibly relevant statistics based on game play so far.
    """

    def __init__(self, rep_trace_forget=0.9, rep_trace_erase=0.7, acc_trace_forget=0.95, acc_trace_erase=0.3, predictor=None):
        self.predictor = predictor
        self.belief_state = BeliefState()
        n_bandits = self.belief_state.n_bandits

        self.action_track = ActionTracker(n_bandits, rep_trace_forget, rep_trace_erase, acc_trace_forget, acc_trace_erase)
        predictor.init_features(self.belief_state, self.action_track)


    def update_step(self, ts, action, result, opp_action):
        """
        Main entry point for updating statistics object. It updates all properties in correct order,
        given observation data from a single step.
        """

        self.belief_state.update_step(ts, action, result, opp_action)
        self.action_track.update_step(ts, action, opp_action)
        self.predictor.update_step(self.belief_state, self.action_track, ts, action, opp_action)


###################################################################################################


class ActionSelectorAgentA():
    """
    Class that uses the available statistics to decide on best action
    """

    def __init__(self, estimator, c_start=0.05, c_mid=0.0, c_end=0.0, midpoint = 1000, n_bandits=100, n_bins=101, 
                trace_select_factor=0.04, tsf_a=1.5, tsf_b=0.5, tsf_c=0.5):
        self.estimator = estimator
        self.pref = np.random.random(n_bandits) * 0.0001 # To break initial ties randomly
        self.c_start = c_start
        self.midpoint = midpoint
        self.c_mid = c_mid
        self.c_end = c_end
        self.n_bandits = n_bandits
        self.trace_select_factor = trace_select_factor
        self.tsf_a = tsf_a
        self.tsf_b = tsf_b
        self.tsf_c = tsf_c

    def choose_action(self):
        e = self.estimator
        b = e.belief_state
        p = e.predictor
        a = e.action_track

        ts = b.ts
        if ts < self.midpoint:
            ratio = ts/self.midpoint
            c = self.c_start * (1.0-ratio) + self.c_mid * ratio
        else:
            ratio = (ts - self.midpoint)/(2000 -  self.midpoint)
            c = self.c_mid * (1.0-ratio) + self.c_end * ratio

        quantile_fraction = c * b.quantile_width

        tsa_schedule = self.tsf_a
        if ts > 599:
            tsa_schedule = self.tsf_b
        if ts > 1199:
            tsa_schedule = self.tsf_c

        opp_trace_adjust = a.opp_rep_trace * self.trace_select_factor * tsa_schedule

        scores = p.expected_via_model + quantile_fraction + opp_trace_adjust + self.pref
        return int(np.argmax(scores))


###################################################################################################


class ActionSelectorAgentB():
    """
    Class that uses the available statistics to decide on best action
    """

    def __init__(self, estimator, factor_c=0.4, follow_trace_p=0.2, c_offset=0.0, n_bandits=100, n_bins=101):
        self.estimator = estimator
        self.pref = np.random.random(n_bandits) * 0.0001 # To break initial ties randomly
        self.factor_c = factor_c
        self.c_offset = c_offset
        self.n_bandits = n_bandits
        self.follow_trace_p = follow_trace_p

    def choose_action(self):
        e = self.estimator
        a = e.action_track
        b = e.belief_state
        p = e.predictor

        c = max(self.factor_c * b.expected.mean() + self.c_offset, 0.0)
        quantile_fraction = c * b.quantile_width

        follow_trace_scale = self.follow_trace_p * b.expected

        trace_follow = a.opp_acc_trace * follow_trace_scale

        scores = p.expected_via_model + quantile_fraction + trace_follow + self.pref

        return int(np.argmax(scores))


###################################################################################################


class CombinedActionSelector():
    """
    Class that uses the available statistics to decide on best action
    """

    def __init__(self, estimator, a, b, ab_switch=500, classifier=None):
        self.estimator = estimator
        self.selector_a = a
        self.selector_b = b
        self.classifier = classifier

        # For normalisation, data source is class_0127
        self.features_mean = np.array([999.3017076562093, 0.3679717453560875, 0.7543197743818979, 2.3445916298014255, 1.4330108332699658, 1.4722086813026856, 0.3733615246534717], dtype=np.float32)
        self.features_std =  np.array([577.1659670826012, 0.10792105527983444, 0.11135945525651045, 0.4291679022706221, 0.5471590428883153, 0.8787078952363906, 0.19254368548804018], dtype=np.float32)

    def calc_features(self):
        e = self.estimator
        a = e.action_track
        b = e.belief_state

        ts = b.ts
        if ts > 0:
            opp_pulls_std = a.n_opp_selections.std()/np.sqrt(ts/101.010101) # Expected 1.0 for uniform random distribution
            pull_diff_std = (a.n_opp_selections - a.n_my_selections).std()/np.sqrt(ts/51.05)
        else:
            opp_pulls_std = 1.0
            pull_diff_std = 1.0

        features = np.array([
            [ts, a.opp_bunching.mean(), a.calc_overlap(), opp_pulls_std, pull_diff_std, 
             a.opponent_mean_trace_a, a.opponent_mean_trace_b]
        ], dtype=np.float32)

        features = (features - self.features_mean)/self.features_std

        return tensor(features)

    def choose_action(self):
        e = self.estimator
        a = e.action_track
        b = e.belief_state

        features = self.calc_features()
        p_choose_b = self.classifier(features).detach().item()
        if np.random.random() < p_choose_b:
            return self.selector_b.choose_action()

        return self.selector_a.choose_action()


###################################################################################################


def agent(observation, configuration):
    global estimator, selector, total_reward, predictor, nn_model_rewards, nn_model_oppclass

    me = observation.agentIndex
    opp = 1 - me
    t = observation.step
    r = observation.reward

    if t == 0:
        predictor = RewardPredictor(nn_model_rewards)
        estimator = CombinedEstimator(predictor=predictor, 
            rep_trace_forget=REP_TRACE_FORGET, rep_trace_erase=REP_TRACE_ERASE,
            acc_trace_forget=ACC_TRACE_FORGET, acc_trace_erase=ACC_TRACE_ERASE)
        
        selector_a = ActionSelectorAgentA(estimator, AGENTA_START_C, AGENTA_MID_C, AGENTA_END_C, 
            midpoint = AGENTA_MIDPOINT, trace_select_factor=AGENTA_TRACE_SELECT_FACTOR,
            tsf_a=AGENTA_TSX, tsf_b=AGENTA_TSY, tsf_c=AGENTA_TSZ)
        selector_b = ActionSelectorAgentB(estimator, AGENTB_C, follow_trace_p=AGENTB_TRACE_SELECT_FACTOR, c_offset=AGENTB_C_OFFSET)
        selector = CombinedActionSelector(estimator, selector_a, selector_b, classifier = nn_model_oppclass)

        total_reward = 0
    else:
        action = observation.lastActions[me]
        opp_action = observation.lastActions[opp]
        last_reward = r - total_reward
        estimator.update_step(t, action, last_reward, opp_action)
        total_reward = r

    action = selector.choose_action()

    return action


####################################################################################################

# reward predictor weights

encoded_weights = '''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
MgBhcmNoaXZlL2RhdGEvMTQwNjYxMzQ3NDIwMjA4RkIuAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlqb+Ym+AMkNP2n/Z76MM8o+tx7QPk0ZRT6qvJ0+W2EBv2mp
ej7gEso+zb6mPoTJzb6Bxas+1/e2vtUsYr7DYYo+WZtavqiAoD4dHAc/gnmvvnLAer6Q7wg+OL+i
vrxOpL40a/2+P7yivmUhSD7UEXw+XMrKvhasYD7A4Uk+C79nvlBLBwiL4VZ2gAAAAIAAAABQSwME
AAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwNjYxMzQ3NzU1NjY0RkIy
AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaEaw/PQgp
lzyn3gU/+isVvv8y0z6xi5m+HAQBv7A1Gb2sk549rsXOvMCpmz3v2MS+K/7svoa+kL34k+O+YG8M
vsQIzz4xSsU81CD5PmfWHT47bme+fdnVPkQZP71aFIw+1eRqvOok2z39sYi9JkmZvEpGa76lmiU9
opH4PVdIVb6OdgS9Ut+TvUFBCT7raT6+22B+PrJEpr7+jrA+Eo5Vvr40I76WMXs9MborPaeYSzy4
0He8quKePjf3Lj70PHc9e+CovoPOEr//pXC+SOzWPURywL5XNUG+0h5ovvOvVz5bwkA+0W+uvDws
lbqnvnu9PlErvkGvK76o75M9n4N8vmZ6cT432Hs+XBeoPZpxJz58LB6+IwKtPQVzyjyP7go7ol6j
uu/MDj6H5fm9fB8hPdVvDr3BC90+q+TtvVharr0RCf88x40QPNbuMr4FjlI+ssjwPU/ZUT2iF5Q9
ASQGPtw6VT5EaBw9IGq0vanqLLzOPPM+SVM7vjy3ND4YQY266uQOvW2ZQL6lmTa+fX4+vsVfFj7W
aKu9DXMavhDIyb0ZqRS+nbKMvatx+r2llvA97Z8fPps3wD0dJRW+h+CEvin4rD2UixW+vDWdvUNU
zT19BUE+IV0SPviqpbzSQks+AhH3vWggHD4wLpA9mLySPd9BhT1ens09tPAsPLIFI77Tozs+NMKI
PjkkpD1fmxY97KSCPnSWy72vAxu+Qw+SPcZYzL7sVu29LjORPj8NDL7smvO9JP+QPCtZvL6X1oe+
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
AzHPvO9e+T1nAwU+Qkqhvp8+Sb2mUsO9ErMjP99LIj5Uxho+vFnRvqxuj77siQk+1jk8PWXyB79j
T4m+BCmevkOCFT7bxqs++UzbPcxAEb7qP6S9ZnCNvrKRTL1tz3u+eImxvljGsby4UIC8KM4EP2WE
9zyHvD+9Z8uPPVz/s75UHJG+6zACv9t5Lr4jvYu+FeUuPq+kGz3D/VY+GicQPpcAxD3fOzm92Ocn
vh6HKL4+Sok+WdmIvacZPL5b43c+7vvGPiObBz9/iqQ9gkfpPTR19L3PIpa9Y+waPb7Vwj3IUls+
IXgyPfZ14rxEw8Y+atbCPUvqgT7hkwa+bSwnvpb33L3LSPs9ZUmEvWlUgT1qtZU+gauYPkd0zb78
fE6+wjYEu7iRozxKCoq+Mx9KPUFaEb4rYhI+8A30vFKmuLz0maA9zkrzPD+L0bw9tga9suE5Pg+4
kD7VrwY+joc9PsHySL6E5JK909KBPhehD76oUnG8XZN5PZxLDD7WoCk+wx9GvoIgk74Ax2g9u269
Pk94Oz1nNQA+VeV/PvzD1roi4cI9pqQQPsAFVD75u2M+W1S9PCbZJr1bFic+a+1bvUzR2T0hRg48
w+BvPsTmMT5Pliy9KZf2vHfO5r0c2Ms9z+hXvnMgHT2vILu8rSnFPZk2Xz46FwG+zbpvvLHf8z2H
ACe+Eom1PSW9rb0aJFA8CEfIveO4kD6yoAC9+/CVvebqmj3hzVg9UqUivv9Y4r01CXW+wjnOvTXq
ojzVMZE+mxYUPjH8Jj7ccyk9/y8HPI7mkzvhme69uRF+vkVhTL38lQY+4kyIvXjpKL6q9Pu94+Wo
vlsj0zxnhm2+lIEvvuQ9Kzy+R5+9CtF2Pb5s3711cto9YYimPt/agz1rUbm+THOnPbYynT6Zx9S8
9AYAvzS4sb0xcv49SvLxPSXibb1MDk++NKsPP26wOr1zddC6trL6PaBgLb6/KJY+N6soPgeZsz2T
L7o+GDtlPpBYgb7+MG+8C2B+Pe76ZL5P5De9er6ZvpiiTz6nI7g+MIS8O1BLBwjT2SOqABAAAAAQ
AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwNjYxMzQ3ODYw
NDE2RkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
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
YWhAPkU0bD83yxO//yOnPgk3AjzT/0I/k5UYPo0hdb/aiKG9dLJ7vp5IPz7wpq4970FYvt1oe7ym
DHy+ctpNvs3yt76oJaW+RzMJP1G7pD1Zp6Q+e0+WPBPBPr7Y2v49eniAvr+rYDyVff++0AIDvzJc
Br8dITo+jUgSP0HZUj521dS+05gJPpidKb2/XdI9wqj+PiY5/j7gLsw91ul5vm0Bmr5oZAe+JOT/
PsF6eD6hXny+SkIOPskl1z3REIG+yplUvr/ZCL8nUEY+1k2Mvja8RLp8e8g9xklUPSmnWT47H3O+
38zjPnZcr72poHu+IqQevgeskj4tS1I+kS/CvOT+bL75IhK+OgDWPdiHQj5fdF68dN0svjesdD3y
sHs+ppssPUr06r3pvbW+usboPRd0RT4DCQ6+Bq+BvkkpOT2nXAs/n0nZPtkGh71oNFa+3C9Rvs8v
6L2wOL2+Mvp9vQ0TD76ldp++Iho0O+Ryzj5z3/Y+xuNkviD0OL2mMlk+fM9MPrit6z5m49a9d+Ya
vf27ED61Bc49pP1TvTUegL4t9F893K3OPevIbz24AjW78FqXvpz9Nz6kVpC+5PkcvVBLBwjD0LmR
ABAAAAAQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwNjYx
MzQ3ODYyNjU2RkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaGLbtPXIcSD6cgyY9gxuxPFoMjz1a4fS9AsEVvhs/8j0TtzS+YvVmvjNnhzys6o08dq99
Pmry271zzi6+7tQxvvI7Oj4wJqk+lz0svojvJj6xd0S+NHxdO/acDzyTaDO+Xy9mvOFhhb2H9CA+
aNpDPrGMib5GjiI+gOpLvmF/X75QSwcIEEcpIIAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAA
AAAAAAAcADYAYXJjaGl2ZS9kYXRhLzE0MDY2MTM0ODAxOTUzNkZCMgBaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWseGAr/0dxq/NwvhPgT6OL53G1c/T2Zq
vlOrTT59a/G+2MwgP/5IpL1gfoq/8ZcIvr3kO76TYww/I8BZPsMctj77LgY+RPBnv+hgXL1QERS9
syIsPv2g7jxVMaG8zvUyP56AnD6lN2s+togtPzfUYr0GvNS+8FL0PLQ6gD4zBBS/UEsHCH4DCzyA
AAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHAA2AGFyY2hpdmUvZGF0YS8xNDA2NjEz
NDgwMzE5MDRGQjIAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpbegk8onB3vV0URb5dyC8+1MNAPgWsBj49ZTo9WNaLvZzKlz0ASXW7DO3vPMQZ173a/Ag+
toQzvebyhr58pLg9LjsuPmd5I77OLH++v2tmva3aHj2bpgW9axLFPDTm/T2f5WE+/ZCWPYRIhj0R
LB4+0z51PFtZ4D1yLw6+7NdDvVBLBwgkUP/rgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAA
AAAAAA8AQwBhcmNoaXZlL3ZlcnNpb25GQj8AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQSwcI0Z5nVQIAAAACAAAAUEsBAgAA
AAAICAAAAAAAAOGHOSjyAwAA8gMAABAAAAAAAAAAAAAAAAAAAAAAAGFyY2hpdmUvZGF0YS5wa2xQ
SwECAAAAAAgIAAAAAAAAnNc30IAFAACABQAAHAAAAAAAAAAAAAAAAABCBAAAYXJjaGl2ZS9kYXRh
LzE0MDY2MTMyOTAyMDY1NlBLAQIAAAAACAgAAAAAAAAhIzf6BAAAAAQAAAAcAAAAAAAAAAAAAAAA
ABAKAABhcmNoaXZlL2RhdGEvMTQwNjYxMzQ3NDE5ODI0UEsBAgAAAAAICAAAAAAAAIvhVnaAAAAA
gAAAABwAAAAAAAAAAAAAAAAAlAoAAGFyY2hpdmUvZGF0YS8xNDA2NjEzNDc0MjAyMDhQSwECAAAA
AAgIAAAAAAAA09kjqgAQAAAAEAAAHAAAAAAAAAAAAAAAAACQCwAAYXJjaGl2ZS9kYXRhLzE0MDY2
MTM0Nzc1NTY2NFBLAQIAAAAACAgAAAAAAADD0LmRABAAAAAQAAAcAAAAAAAAAAAAAAAAABAcAABh
cmNoaXZlL2RhdGEvMTQwNjYxMzQ3ODYwNDE2UEsBAgAAAAAICAAAAAAAABBHKSCAAAAAgAAAABwA
AAAAAAAAAAAAAAAAkCwAAGFyY2hpdmUvZGF0YS8xNDA2NjEzNDc4NjI2NTZQSwECAAAAAAgIAAAA
AAAAfgMLPIAAAACAAAAAHAAAAAAAAAAAAAAAAACQLQAAYXJjaGl2ZS9kYXRhLzE0MDY2MTM0ODAx
OTUzNlBLAQIAAAAACAgAAAAAAAAkUP/rgAAAAIAAAAAcAAAAAAAAAAAAAAAAAJAuAABhcmNoaXZl
L2RhdGEvMTQwNjYxMzQ4MDMxOTA0UEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAA
AAAAAAAAkC8AAGFyY2hpdmUvdmVyc2lvblBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAoAAAAAAAAA
CgAAAAAAAADLAgAAAAAAABIwAAAAAAAAUEsGBwAAAADdMgAAAAAAAAEAAABQSwUGAAAAAAoACgDL
AgAAEjAAAAAA
'''

decoded = base64.b64decode(encoded_weights)
buffer = io.BytesIO(decoded)
nn_model_rewards.load_state_dict(torch.load(buffer))


############################################################################

# opposition classifier weights

encoded_weights = '''UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpa
WlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAShYDAAAAG5ldC4wLndlaWdo
dHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMoKFgHAAAAc3RvcmFnZXEEY3Rv
cmNoCkZsb2F0U3RvcmFnZQpxBVgPAAAAMTQwMjQwMTIzMTM0OTI4cQZYAwAAAGNwdXEHS3B0cQhR
SwBLEEsHhnEJSwdLAYZxColoAClScQt0cQxScQ1YCgAAAG5ldC4wLmJpYXNxDmgDKChoBGgFWA8A
AAAxNDAyNDAxMjI3NTQzNTJxD2gHSxB0cRBRSwBLEIVxEUsBhXESiWgAKVJxE3RxFFJxFVgMAAAA
bmV0LjIud2VpZ2h0cRZoAygoaARoBVgPAAAAMTQwMjQwMDczODExNTM2cRdoB00AAXRxGFFLAEsQ
SxCGcRlLEEsBhnEaiWgAKVJxG3RxHFJxHVgKAAAAbmV0LjIuYmlhc3EeaAMoKGgEaAVYDwAAADE0
MDI0MDEyMjYzNzkzNnEfaAdLEHRxIFFLAEsQhXEhSwGFcSKJaAApUnEjdHEkUnElWAwAAABuZXQu
NC53ZWlnaHRxJmgDKChoBGgFWA8AAAAxNDAyNDAxMjI2NDMxMzZxJ2gHTQABdHEoUUsASxBLEIZx
KUsQSwGGcSqJaAApUnErdHEsUnEtWAoAAABuZXQuNC5iaWFzcS5oAygoaARoBVgPAAAAMTQwMjQw
MTIzMTU2ODMycS9oB0sQdHEwUUsASxCFcTFLAYVxMoloAClScTN0cTRScTVYDAAAAG5ldC42Lndl
aWdodHE2aAMoKGgEaAVYDwAAADE0MDI0MDEyMzE1OTAwOHE3aAdLEHRxOFFLAEsBSxCGcTlLEEsB
hnE6iWgAKVJxO3RxPFJxPVgKAAAAbmV0LjYuYmlhc3E+aAMoKGgEaAVYDwAAADE0MDI0MDEyMzE1
OTA4OHE/aAdLAXRxQFFLAEsBhXFBSwGFcUKJaAApUnFDdHFEUnFFdX1xRlgJAAAAX21ldGFkYXRh
cUdoAClScUgoWAAAAABxSX1xSlgHAAAAdmVyc2lvbnFLSwFzWAMAAABuZXRxTH1xTWhLSwFzWAUA
AABuZXQuMHFOfXFPaEtLAXNYBQAAAG5ldC4xcVB9cVFoS0sBc1gFAAAAbmV0LjJxUn1xU2hLSwFz
WAUAAABuZXQuM3FUfXFVaEtLAXNYBQAAAG5ldC40cVZ9cVdoS0sBc1gFAAAAbmV0LjVxWH1xWWhL
SwFzWAUAAABuZXQuNnFafXFbaEtLAXN1c2IuUEsHCMDjunndAwAA3QMAAFBLAwQAAAgIAAAAAAAA
AAAAAAAAAAAAAAAAHAAZAGFyY2hpdmUvZGF0YS8xNDAyNDAwNzM4MTE1MzZGQhUAWlpaWlpaWlpa
WlpaWlpaWlpaWlpaO6zTPiH1uT5+VJ0+fdYVvPKPx75Ipxg/M08tvmnS8L71Ioc9Ao8Cv6L/Pr+F
GhS9JVE3Pv9sTTzg98e+1PfhPtLbUb571t684gIHv0uRib7JSx0/j7tLPq2eCb8Xnxq/OwhjPlNI
L70Roi4/iKLgvqXq7j78dD8/XCycvYN+7r4vlwK/4W2UO3F99D7FYWc9PF6/Pj4Lkj3lw10+22mq
vnbEOb6zuVO+VDnbPQENCT+tkro+VpUePKftnb0aLd+95qohv5CCH7/7ne8+pKaIPmLZcz64ZkK+
c/QePzyR6z3IAkE/Lho0Pl1Bs74f/C89s69VPjzeAj4cdYe+SH28vqnNCb8VjoW+IZ+wPo001j0b
EgG/ZwcCvwhDuT61Gse+AA5RPiT/Mb/wmZ2+i1d1PSYyFD7g1zi+A4rYvfh25758JiQ+V4cGvx98
Wb4ZE5y+QUEfPTYXxz0Z9im+WJ8OPHTTH7+PKvs+6sibvbFMVj5YFCS/pKVgvs9nI79a2IY9LxFm
PrzORL2I3Oe+L8sEv3sNKr3E6C4/6qUWPQ6HqD6QR6o9OFHVOug2iL1I+/I985lBvhO5Sb57G8S+
NrvRvjMB8z51WQS/Wm2rvubdC78955o+KVjivL4CMz3yKSO/S6f5vm5MTb+S3WU/ge1qvy8+yT24
qdo+BeYDPitB2z6INQS/rSEBPuiUsr2dIGU+7mQEPwhOQr6niBs/nJ0Vv5Bksj6xPGo+wGkTPt1p
CD+Sy2u8oiNsvXORAz6TCpy+Djs+vyJihr43Tdg+xh9FPv1lFD72hI6+fD14P3kz8T6i37u9Fa64
vtFJ2j1FYfM+6XA4P90pGD6/NTy+xCojPrDFgD56OKQ+Cm1Bv4L8g73DWhK+yqnHPsTpJb6JF927
/1qxPtZLBj9KvWy9vL7jvow5jb5nKjA97cN5vrAhlL5/YKm+KtUxPiLrED9taKG+OT8TvgwmHb//
HKc+HH7xvVGB+b5S1xC/2q/tvVp3XD5rhG4/kA1GvpH1Ij9fvhs/Z4CLPmXNxL7vnle//ER3vjyP
BT/Nm7w+Ew8Tvl8XiL5yQBU/QdcLPtUplz70Dx8/1RqNv4qW+b4vp42/V3JHv1KGCb638Dg97cE+
PuhIAz9UAUG/8dLXPmyyxj7UPeI+Fur9PS0DHz8D+Fq/NlUAP3eqH7+ZRiu/YhUMvmG4Kr6895G+
ujz6voC4vz2isgI/SWgQPt33sL4WYJ4+VIouv4JP3z32Ij4+qrYrPvZzBz+ORhW+PYJ3vWCjLj2f
Ac68mgGGvlvMQD1Pdr6+0UewvS3uTL3maVe/G1a5Pb02Mb9WzSK+GoOLPuT8iD4hFAo/Ox39viE7
YT3iKZU9JSscvVBLBwit3GJcAAQAAAAEAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBh
cmNoaXZlL2RhdGEvMTQwMjQwMTIyNjM3OTM2RkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa8lKcPfIOxLwia0i8FxQGPyjxAz7/TCu/e4w8vtYZlb64
51e+P0fwvTm7LD0MNYK+GfAxPtSJ1b0xtNY9FrDaPlBLBwgMSxwHQAAAAEAAAABQSwMEAAAICAAA
AAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwMjQwMTIyNjQzMTM2RkIyAFpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaSbVdPzB0Hr+L0V2+
4XMFvyPMC78v1+s+oe0hP2c8Rb9ArEa/s48Wv8dwyT4zfZy+K+q/PgYamz6nyAy/PXJGv5JXa78Z
eAs/VosoP01w2D6Z9Pk+YfkCv829QL8qWDs/9xViP93SLD/9nP29ekINPx62S75Swmi+V8LxPkTb
Ij9MWmu+Xa5YP68W2j6p3A8/Nm9ZP1vT6L6v9ea+jRPpPkVaJD+YbBM/7PZAvsdX+j7U2xi/39rf
vllMMz/BmWs/2F0bv+GBPj9IigA/m1YaPybq9D7MoA6/qfx/vzgBDj9WPyk/pCsaP3KP6b57M4g+
K+VavuY4t77csPc+u6ZAPw9kDT+lZQC/33+PvojfTL/0dU2/NfcwP9X1MD/w4g6/NFc8v58q4r7A
9E48sIMmv+aEKz8XUDU/tJ0Fv6oiJb/IvQE/B/Ilv48OGb+DoOe+SXkDv73NNz8zmx8/i/Afvw+N
6b6Mciu/fQKNPbFE4b4eL3Y+QsurPgy9475AjhW/jiD0vqPk7T6X1Qc/04Q0P0jVrD4o4eW+U+WJ
v5crLz9VVn4/uMmPP11MlL6UOpY+JCNKvYXwar7Exkc/NhkHP9Xf9D5jZGy/swOJvpscYr/MKTS/
+c2OPyAYbz/MMBK/jw8gvxjOmb67U+U+qfIKv5H9AT/idwQ/NlWQvvolDb/1a0k/Bxozv9GsAr/9
9j6/ztG+vlv0uz5l42c/eGc5v1IbN79eeCO/0qD6PkQp374M6IA97YZpPlDNAr9JEA6/fJxiP0De
Lr+unVa/WcLDviUo3L5arM8+tKctP3cKhL5a+eO+RSFKv9s5tz6hgLW+e6g+PhnESj5Qi7a+5ZAP
v6RmR76MYRQ/Su4rP6hehz9zIic/mMzZv/BQoL/v4hc/nHYlP7phHj9JV8a8Ob2iPhvsNL+RfBi/
tlmOPnfB/T7zeQe+xlAiPxR6PT8wGCo/Q84pP95QML+kggu/Kd0aP0dU2D71hA8/JwgavjPmDj8o
E52+USo+vwcnKj9TalM/kbT1vimALT/YDmk+JvjnPi5jFj/ANBW/mzgUv+y35D5L5Sw/bp8TP5Fp
9r6j/bU+GgucvkVN6L6FTAI/O4BVP0CjR7/GLjo/NgTwPU8o9j7JDig/pSXcvnELSr/8RTM/gO0Y
PyFeDz82xAS/UHwFPplZib6CngW/cawEP/vIQD8ZK1E/zfFhv773k7wZLYi+pHYuvwjw1j6b/mQ/
amxHvxS97r40uAW/nBwMP8TPw715SA8+iyqVPtpRIb9zZOq+HJw/v2EZJz++Vjs/37ofP+JZND85
vNW++jMsv3+mED9Lilg/qIUfP4CVpr7LIiU//7NlvlYGCL8/2Tc+UgRdP1BLBwianPK5AAQAAAAE
AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwMjQwMTIyNzU0
MzUyRkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
cBG6vlFBpD7q4AC+LfuwPiX6D79I2gO/NxTyvv7Ntz7meCE+qJ+UPgse7L6MAv67CphLvm9n7r6D
axE/rxg2v1BLBwjOZu2uQAAAAEAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNo
aXZlL2RhdGEvMTQwMjQwMTIzMTM0OTI4RkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaTFmwPljxkj5RFhA/nkyIvVjFpD4MdHA790YXv345xL0v9w+/
fwhgv+kCtrzLcfe+ttUbPqr49r5viyA/MjKkPch6wT1aOzG+k1IsPv94EL8CgrW9z8IhP3S01r2n
AUo9B6VmvYTVN76WGLu+a9X9P6W3d75Tu0Y/BrM6vsdcAD+c9iW9ps8Hv+RiNL8qG46+KJYvvjiK
Y72GoQC9MlMnvbSvwrz9oSy+zmlfPgdwzL57EAK+1AlRvu43fb1aoIY9IQNKP/W2or5MlsS9f0ow
vU4dub4Pc2k+ay/jvfevIj8k2Da/sZsuP4giQL9+KA0/EGFWv6yLcDsMP5g/sjbSPl28Tz+V7aq+
croGP0Q3HT72b3i/4YX6PsM03j0Pgoo+vBqnPsbXND7EAq8+/gv+PaR0d79f1k0/CLs9vkeQe71L
zrY+QQVovJ542L4yLEE/KN45v9Vl7b7ScTE+8AcXv+b9zz693om+z4Q5PZcxE7+YPIq+rVWTvlzx
yr7Oe6m+BVwWP6uDM79gD2Y9OzKkvb+zgT3pTbu+qZwlPvmzYD5Rt2a/RdrvPsXmGr6sY8g+5wcn
v83KyL6mu9W+xKm2PVBLBwh/5A+xwAEAAMABAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwA
NgBhcmNoaXZlL2RhdGEvMTQwMjQwMTIzMTU2ODMyRkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaCc5BvuF9dT7Fm58+lJD9PRG+y74cygO/F6fNPjJD
2L6ho5G+3k+XvDYjCj/s0x8/FOQVPhbXYT5FfZi+j2eFPlBLBwh8VDSPQAAAAEAAAABQSwMEAAAI
CAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEvMTQwMjQwMTIzMTU5MDA4RkIyAFpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa+r8KwEf0E0Dc
dBpAAmg5QAtR6L8QwQDAp1IoQJevyb/g0gHAzm4fwB643T8xzdM/du2SQC81EUA0bRTA3fo1QFBL
BwgJA1ajQAAAAEAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABwANgBhcmNoaXZlL2RhdGEv
MTQwMjQwMTIzMTU5MDg4RkIyAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlparEYqPVBLBwjqCO1lBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAA
AA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa
WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwECAAAAAAgIAAAA
AAAAwOO6ed0DAADdAwAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAA
CAgAAAAAAACt3GJcAAQAAAAEAAAcAAAAAAAAAAAAAAAAAC0EAABhcmNoaXZlL2RhdGEvMTQwMjQw
MDczODExNTM2UEsBAgAAAAAICAAAAAAAAAxLHAdAAAAAQAAAABwAAAAAAAAAAAAAAAAAkAgAAGFy
Y2hpdmUvZGF0YS8xNDAyNDAxMjI2Mzc5MzZQSwECAAAAAAgIAAAAAAAAmpzyuQAEAAAABAAAHAAA
AAAAAAAAAAAAAABQCQAAYXJjaGl2ZS9kYXRhLzE0MDI0MDEyMjY0MzEzNlBLAQIAAAAACAgAAAAA
AADOZu2uQAAAAEAAAAAcAAAAAAAAAAAAAAAAANANAABhcmNoaXZlL2RhdGEvMTQwMjQwMTIyNzU0
MzUyUEsBAgAAAAAICAAAAAAAAH/kD7HAAQAAwAEAABwAAAAAAAAAAAAAAAAAkA4AAGFyY2hpdmUv
ZGF0YS8xNDAyNDAxMjMxMzQ5MjhQSwECAAAAAAgIAAAAAAAAfFQ0j0AAAABAAAAAHAAAAAAAAAAA
AAAAAADQEAAAYXJjaGl2ZS9kYXRhLzE0MDI0MDEyMzE1NjgzMlBLAQIAAAAACAgAAAAAAAAJA1aj
QAAAAEAAAAAcAAAAAAAAAAAAAAAAAJARAABhcmNoaXZlL2RhdGEvMTQwMjQwMTIzMTU5MDA4UEsB
AgAAAAAICAAAAAAAAOoI7WUEAAAABAAAABwAAAAAAAAAAAAAAAAAUBIAAGFyY2hpdmUvZGF0YS8x
NDAyNDAxMjMxNTkwODhQSwECAAAAAAgIAAAAAAAA0Z5nVQIAAAACAAAADwAAAAAAAAAAAAAAAADU
EgAAYXJjaGl2ZS92ZXJzaW9uUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAACgAAAAAAAAAKAAAAAAAA
AMsCAAAAAAAAUhMAAAAAAABQSwYHAAAAAB0WAAAAAAAAAQAAAFBLBQYAAAAACgAKAMsCAABSEwAA
AAA=
'''

decoded = base64.b64decode(encoded_weights)
buffer = io.BytesIO(decoded)
nn_model_oppclass.load_state_dict(torch.load(buffer))