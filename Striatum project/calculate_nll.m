function neg_log_likelihood = calculate_nll(params_to_fit, session_data)
% CALCULATE_NLL - Calculates the negative log-likelihood of behavioral data
% given a set of Q-learning model parameters.
%
% INPUTS:
%   params_to_fit: A vector containing the model parameters to be fitted.
%       [alpha, beta, initial_sigma, sigma_decay, min_sigma]
%   session_data: A struct array where each element is a trial, containing
%       the fields 'states' and 'actions'.
%
% OUTPUT:
%   neg_log_likelihood: A single scalar value representing the total
%       negative log-likelihood of the session data given the parameters.

%% 0. --- NEW: Make Function Deterministic ---
% Reset the random number generator stream. This is CRITICAL. It ensures
% that for the same set of input parameters, the sequence of "random"
% perceptual noise is identical on every function call, which is a
% requirement for the fmincon optimizer.
rng(1, 'twister');

%% 1. Unpack Parameters and Initialize Model
% Parameters to be fit
alpha       = params_to_fit(1);
beta        = params_to_fit(2);
initial_sigma = params_to_fit(3);
sigma_decay = params_to_fit(4);
min_sigma   = params_to_fit(5);

% Fixed model parameters
gamma = 0.95; % Discount factor
env.corridorLength = 200;
rewards.inZone = 20;
rewards.moveCost = -0.2;
rewards.lickCost = -0.5;
env.rewardZoneStart = 101;
env.rewardZoneEnd = 125;
actions.moveForward = 1;
actions.lick = 2;
numActions = 2;

% Initialize model state
total_log_likelihood = 0;
q_table = zeros(env.corridorLength, numActions);
current_sigma = initial_sigma;

%% 2. Loop Through All Trials and Timesteps
num_trials = length(session_data);
for trial_idx = 1:num_trials
    
    trial_states = session_data(trial_idx).states;
    trial_actions = session_data(trial_idx).actions;
    
    % Flag for one reward per visit
    reward_collected_this_visit = false;

    for t = 1:length(trial_states)
        
        % The state and action the ANIMAL was in/took
        true_s = trial_states(t);
        action_taken = trial_actions(t);
        
        % The agent's perception is noisy
        noise = current_sigma * randn();
        perceived_s = round(true_s + noise);
        perceived_s = max(1, min(env.corridorLength, perceived_s));

        % --- LIKELIHOOD CALCULATION ---
        % A. Get the model's current Q-values for the perceived state
        q_values_for_state = q_table(perceived_s, :);
        
        % B. Use softmax to convert Q-values to action probabilities
        %    (Subtracting max for numerical stability doesn't change result)
        q_vals_stable = q_values_for_state - max(q_values_for_state);
        action_probs = exp(beta * q_vals_stable) / sum(exp(beta * q_vals_stable));
        
        % C. Find the probability of the action the animal *actually* took
        prob_of_observed_action = action_probs(action_taken);
        
        % D. Add its log to the total (add a small epsilon for stability)
        total_log_likelihood = total_log_likelihood + log(prob_of_observed_action + 1e-9);

        % --- MODEL UPDATE ---
        % E. Determine the reward and next state from the animal's choice
        reward = 0;
        next_true_s = true_s;
        
        switch action_taken
            case actions.moveForward
                next_true_s = true_s + 1;
                reward = rewards.moveCost;
            case actions.lick
                reward = rewards.lickCost;
                if true_s >= env.rewardZoneStart && true_s <= env.rewardZoneEnd
                    if ~reward_collected_this_visit
                        reward = reward + rewards.inZone;
                        reward_collected_this_visit = true;
                    end
                end
        end

        if true_s < env.rewardZoneStart || true_s > env.rewardZoneEnd
             reward_collected_this_visit = false;
        end
        
        % F. Update the Q-table just as our simulation would
        is_terminal = (next_true_s >= env.corridorLength);
        
        if ~is_terminal
            % Perception of next state is also noisy
            next_noise = current_sigma * randn();
            next_perceived_s = round(next_true_s + next_noise);
            next_perceived_s = max(1, min(env.corridorLength, next_perceived_s));
            
            max_next_q = max(q_table(next_perceived_s, :));
            td_target = reward + gamma * max_next_q;
        else
            td_target = reward; % No future rewards from terminal state
        end
        
        current_q = q_table(perceived_s, action_taken);
        td_error = td_target - current_q;
        
        q_table(perceived_s, action_taken) = current_q + alpha * td_error;
    end
    
    % G. Decay sigma at the end of the trial
    current_sigma = max(min_sigma, current_sigma * sigma_decay);
end

% 3. Return the NEGATIVE of the log likelihood for minimization
neg_log_likelihood = -total_log_likelihood;

end
