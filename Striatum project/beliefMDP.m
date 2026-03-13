% BELIEF-MDP AGENT - CORRECTED VERSION

clc; close all;

% --- Hyperparameters ---
MAX_EPISODES = 200;
ALPHA = 0.03;
GAMMA = 0.99;
LAMBDA = 0.9;
EPSILON = 0.3;
EPSILON_DECAY = 0.995;
MIN_EPSILON = 0.02;

% --- Environment & Belief ---
env = LinearTrackPOMDP();
noise_params = env.get_noise_params();

% Increase drift for meaningful uncertainty
env.ProprioNoiseRate = 0.15;  % Higher drift
noise_params.proprio_rate = 0.15;

belief = BeliefState(noise_params);

% --- RBF Setup (FIXED NORMALIZATION) ---
% Belief μ: 0-200cm → normalize by 200 → [0, 1]
% Belief σ: typically 1-15cm → normalize by 15 → [0, 1]
SIGMA_NORMALIZER = 20;  % Not 30!

n_mu_centers = 15;
n_sigma_centers = 8;

[C1, C2] = meshgrid(linspace(-0.05, 1.1, n_mu_centers), ...
                    linspace(0, 1.2, n_sigma_centers));
RBF_CENTERS = [C1(:), C2(:)];
NUM_RBF = size(RBF_CENTERS, 1);
RBF_SIGMA = 0.1;

% --- Action Space ---
ActionSet = [
    0,  0;   % Stop
    20, 0;   % Walk
    40, 0;   % Jog  
    60, 0;   % Sprint
    30, 1;   % Lick moving
    15, 1;   % Lick slow
    0,  1;   % Lick stop
];
NUM_ACTIONS = size(ActionSet, 1);

% --- Weights (with small optimistic initialization) ---
Weights = ones(NUM_RBF, NUM_ACTIONS) * 0.1;

% --- Analysis Storage ---
NUM_BINS = 50;
BIN_EDGES = 0:4:200;
BIN_CENTERS = BIN_EDGES(1:end-1) + 2;

all_trials_vel = nan(MAX_EPISODES, NUM_BINS);
all_trials_lick = nan(MAX_EPISODES, NUM_BINS);
all_trials_belief_mu = nan(MAX_EPISODES, NUM_BINS);
all_trials_belief_sigma = nan(MAX_EPISODES, NUM_BINS);
all_trials_p_in_rz = nan(MAX_EPISODES, NUM_BINS);
all_trials_error = nan(MAX_EPISODES, NUM_BINS);

episode_rewards = zeros(1, MAX_EPISODES);

%% Training Loop
for episode = 1:MAX_EPISODES
    
    raw_obs = env.reset();
    belief.reset();
    
    % Initial belief features
    [b_mu, b_sigma] = get_belief_features(belief, SIGMA_NORMALIZER);
    phi = compute_rbf([b_mu; b_sigma], RBF_CENTERS, RBF_SIGMA);
    q_values = phi' * Weights;
    
    % Action selection (epsilon-greedy)
    if rand < EPSILON
        action_idx = randi(NUM_ACTIONS);
    else
        [~, action_idx] = max(q_values);
    end
    
    % Eligibility traces
    E_traces = zeros(NUM_RBF, NUM_ACTIONS);
    
    done = false;
    ep_reward = 0;
    
    % Trajectory storage (pre-allocate for speed)
    max_steps = 2500;
    traj_pos = nan(1, max_steps);
    traj_vel = nan(1, max_steps);
    traj_lick = nan(1, max_steps);
    traj_belief_mu = nan(1, max_steps);
    traj_belief_sigma = nan(1, max_steps);
    traj_p_rz = nan(1, max_steps);
    traj_error = nan(1, max_steps);
    
    step_count = 0;
    
    while ~done && step_count < max_steps
        step_count = step_count + 1;
        chosen_action = ActionSet(action_idx, :);
        
        % --- Record PRE-step state ---
        traj_pos(step_count) = env.Position;
        traj_vel(step_count) = env.CurrentVelocity;
        traj_lick(step_count) = chosen_action(2);
        traj_belief_mu(step_count) = belief.mu;
        traj_belief_sigma(step_count) = sqrt(belief.sigma2);
        traj_p_rz(step_count) = belief.prob_in_reward_zone(noise_params.reward_zone);
        traj_error(step_count) = belief.mu - env.Position;
        
        % --- Step environment ONCE ---
        [raw_obs, reward, done, info] = env.step(chosen_action);
        
        % --- Belief Update ---
        belief.update(raw_obs);
        
        % --- Next state features ---
        [b_mu, b_sigma] = get_belief_features(belief, SIGMA_NORMALIZER);
        next_phi = compute_rbf([b_mu; b_sigma], RBF_CENTERS, RBF_SIGMA);
        next_q_values = next_phi' * Weights;
        
        % Next action (SARSA - on-policy)
        if rand < EPSILON
            next_action_idx = randi(NUM_ACTIONS);
        else
            [~, next_action_idx] = max(next_q_values);
        end
        
        % --- TD Error ---
        current_q = q_values(action_idx);
        next_q = next_q_values(next_action_idx);
        td_target = reward + GAMMA * next_q * (~done);
        td_error = td_target - current_q;
        
        % --- Eligibility Trace Update (replacing traces for stability) ---
        E_traces = E_traces * GAMMA * LAMBDA;
        E_traces(:, action_idx) = phi;  % Replacing trace
        
        % --- Weight Update ---
        Weights = Weights + ALPHA * td_error * E_traces;
        
        % Advance
        phi = next_phi;
        q_values = next_q_values;
        action_idx = next_action_idx;
        ep_reward = ep_reward + reward;
    end
    
    % Trim trajectory
    traj_pos = traj_pos(1:step_count);
    traj_vel = traj_vel(1:step_count);
    traj_lick = traj_lick(1:step_count);
    traj_belief_mu = traj_belief_mu(1:step_count);
    traj_belief_sigma = traj_belief_sigma(1:step_count);
    traj_p_rz = traj_p_rz(1:step_count);
    traj_error = traj_error(1:step_count);
    
    % Decay exploration
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY);
    episode_rewards(episode) = ep_reward;
    
    % --- Spatial Binning ---
    bin_idx = discretize(traj_pos, BIN_EDGES);
    valid = ~isnan(bin_idx);
    
    if any(valid)
        bi = bin_idx(valid);
        all_trials_vel(episode, :) = accumarray(bi', traj_vel(valid)', [NUM_BINS 1], @mean, NaN);
        all_trials_lick(episode, :) = accumarray(bi', traj_lick(valid)', [NUM_BINS 1], @mean, NaN);
        all_trials_belief_mu(episode, :) = accumarray(bi', traj_belief_mu(valid)', [NUM_BINS 1], @mean, NaN);
        all_trials_belief_sigma(episode, :) = accumarray(bi', traj_belief_sigma(valid)', [NUM_BINS 1], @mean, NaN);
        all_trials_p_in_rz(episode, :) = accumarray(bi', traj_p_rz(valid)', [NUM_BINS 1], @mean, NaN);
        all_trials_error(episode, :) = accumarray(bi', traj_error(valid)', [NUM_BINS 1], @mean, NaN);
    end
    
    % Progress
    if mod(episode, 100) == 0
        recent_reward = mean(episode_rewards(max(1,episode-50):episode));
        fprintf('Episode %d | Reward: %.1f (avg: %.1f) | ε: %.3f\n', ...
            episode, ep_reward, recent_reward, EPSILON);
    end
end

%% Visualization
figure('Name', 'Belief-MDP Results (Fixed)', 'Position', [50 50 1400 900]);

early_idx = 1:30;
late_idx = (MAX_EPISODES-49):MAX_EPISODES;

% 1. Learning curve
subplot(2,4,1);
plot(smooth(episode_rewards, 30), 'k', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Total Reward');
title('Learning Curve'); grid on;

% 2. Velocity profile (early vs late)
subplot(2,4,2); hold on;
plot_shaded(BIN_CENTERS, all_trials_vel(early_idx, :), [0.7 0.7 0.7]);
plot_shaded(BIN_CENTERS, all_trials_vel(late_idx, :), [0 0 0]);
xline(100, 'g--', 'LineWidth', 1.5); xline(125, 'g--', 'LineWidth', 1.5);
xlabel('Position (cm)'); ylabel('Velocity (cm/s)');
title('Velocity Profile'); legend('Early', 'Late');

% 3. Lick profile
subplot(2,4,3); hold on;
plot_shaded(BIN_CENTERS, all_trials_lick(early_idx, :), [1 0.7 0.7]);
plot_shaded(BIN_CENTERS, all_trials_lick(late_idx, :), [0.8 0 0]);
xline(100, 'g--', 'LineWidth', 1.5); xline(125, 'g--', 'LineWidth', 1.5);
xlabel('Position (cm)'); ylabel('P(lick)');
title('Lick Profile'); legend('Early', 'Late');

% 4. Belief accuracy
subplot(2,4,4);
mu_late = nanmean(all_trials_belief_mu(late_idx, :), 1);
plot(BIN_CENTERS, BIN_CENTERS, 'k--', 'LineWidth', 1); hold on;
plot(BIN_CENTERS, mu_late, 'b', 'LineWidth', 2);
xlabel('True Position (cm)'); ylabel('Belief μ (cm)');
title('Belief Accuracy'); legend('Perfect', 'Agent', 'Location', 'NW');
axis equal; xlim([0 200]); ylim([0 200]);

% 5. Positional uncertainty
subplot(2,4,5); hold on;
plot_shaded(BIN_CENTERS, all_trials_belief_sigma(early_idx, :), [0.7 0.7 1]);
plot_shaded(BIN_CENTERS, all_trials_belief_sigma(late_idx, :), [0 0 0.8]);
xline(80, 'm--', 'LineWidth', 1.5);
xline(100, 'g--', 'LineWidth', 1.5);
xlabel('Position (cm)'); ylabel('Belief σ (cm)');
title('Positional Uncertainty');

% 6. P(in reward zone)
subplot(2,4,6);
plot_shaded(BIN_CENTERS, all_trials_p_in_rz(late_idx, :), [0 0.6 0]);
xline(100, 'g--', 'LineWidth', 1.5); xline(125, 'g--', 'LineWidth', 1.5);
xlabel('Position (cm)'); ylabel('P(in RZ)');
title('Belief: P(position ∈ RZ)');

% 7. Belief error
subplot(2,4,7);
plot_shaded(BIN_CENTERS, all_trials_error(late_idx, :), [0.5 0 0.5]);
yline(0, 'k--');
xline(80, 'm--'); xline(100, 'g--');
xlabel('Position (cm)'); ylabel('Belief μ - True Pos (cm)');
title('Belief Error');

% 8. Value function in belief space
subplot(2,4,8);
visualize_belief_value(Weights, RBF_CENTERS, RBF_SIGMA, noise_params.reward_zone);
xlabel('Belief μ / 200'); ylabel('Belief σ / 15');
title('Value Function V(b)');

sgtitle('Belief-MDP Agent (Corrected)', 'FontSize', 14);

%% Helper Functions

function [b_mu, b_sigma] = get_belief_features(belief, sigma_norm)
    b_mu = belief.mu / 200;
    b_sigma = sqrt(belief.sigma2) / sigma_norm;
    
    % Soft clipping for stability
    b_mu = max(-0.1, min(1.2, b_mu));
    b_sigma = max(0, min(1.5, b_sigma));
end

function phi = compute_rbf(state, centers, sigma)
    diffs = centers - state';
    sq_dists = sum(diffs.^2, 2);
    phi = exp(-sq_dists / (2 * sigma^2));
    
    % Normalize for stability
    phi = phi / (sum(phi) + 1e-8);
end

function visualize_belief_value(W, centers, sigma, rz)
    [xx, yy] = meshgrid(0:0.02:1.1, 0:0.02:1.2);
    V = zeros(size(xx));
    for i = 1:numel(xx)
        phi = compute_rbf([xx(i); yy(i)], centers, sigma);
        phi = phi / (sum(phi) + 1e-8);
        V(i) = max(phi' * W);
    end
    imagesc([0 1.1], [0 1.2], V); axis xy; colorbar;
    hold on;
    % Mark RZ location in belief space
    xline(rz(1)/200, 'g--', 'LineWidth', 2);
    xline(rz(2)/200, 'g--', 'LineWidth', 2);
end

function plot_shaded(x, data, color)
    mu = nanmean(data, 1);
    sem = nanstd(data, [], 1) ./ sqrt(sum(~isnan(data), 1));
    mu = fillmissing(mu, 'linear');
    sem = fillmissing(sem, 'linear');
    fill([x fliplr(x)], [mu+sem fliplr(mu-sem)], color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    plot(x, mu, 'Color', color, 'LineWidth', 2);
end