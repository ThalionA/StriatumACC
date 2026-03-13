%% Simulate_POMDP_Striatum_v2.m
% Forward simulation of a belief-state RL agent in a spatial corridor.
% Features: Average reward continuous TD learning, velocity-coupled 
% action costs, and emergent Kalman map stabilization.
clear; clc; close all;

%% 1. Task & Kinematic Parameters
corridor_end   = 200; 
visual_landmark = 80;
reward_start   = 100;
reward_end     = 135;
dx             = 2; 
x_space        = 0:dx:corridor_end;
N_bins         = length(x_space);
N_trials       = 200;

% Kinematics
v_run          = 30;   % cm/s (Running velocity)
v_lick         = 10;   % cm/s (Velocity when licking/engaged)

%% 2. Agent Parameters
% --- Kalman Filter (Emergent Map Stabilization) ---
Q_noise        = 0.5;  % Process noise variance per cm traveled
R_slope        = 0.2;  % Spatial scaling of visual observation noise
R_min          = 2.0;  % Minimum observation variance at the landmark
ITI_inflation  = 5.0;  % Variance injected between trials (forgetting)

% --- RL Parameters (Average Reward TD) ---
eta_w          = 0.05; % Learning rate for spatial value weights
eta_rho        = 0.01; % Learning rate for average reward
gamma          = 1.0;  % Undiscounted formulation (handled by rho)
beta           = 4;    % Softmax inverse temperature
initial_rho    = 0.01; % Initial estimate of global reward rate

%% 3. Initialize Variables
w_val               = zeros(1, N_bins); % State-value weights
rho                 = initial_rho;      % Global average reward rate
Latents.Uncertainty = zeros(N_trials, N_bins);
Latents.Value       = zeros(N_trials, N_bins);
Latents.RPE         = zeros(N_trials, N_bins);
Behavior.Licks      = zeros(N_trials, N_bins);
Behavior.Velocity   = zeros(N_trials, N_bins);

% Initial spatial uncertainty
Sigma_t = 50; 

%% 4. Simulation Loop
for trial = 1:N_trials
    
    % Inter-trial interval increases uncertainty slightly, but overall 
    % variance will decay over trials mechanistically via the Kalman gain.
    Sigma_t = Sigma_t + ITI_inflation; 
    mu_t    = 0; 
    reward_collected = false;
    
    % Generate initial belief state
    b_t = exp(-0.5 * ((x_space - mu_t).^2) / max(Sigma_t, 1e-3));
    b_t = b_t / sum(b_t);
    
    for t = 1:N_bins-1
        true_x = x_space(t);
        
        % --- A. VALUE & POLICY ---
        % Current state value
        V_t = sum(w_val .* b_t);
        Latents.Value(trial, t) = V_t;
        
        % Action Selection (Lick probability scales with state value)
        P_lick = 1 / (1 + exp(-beta * (V_t - 0.2))); % 0.2 is baseline threshold
        did_lick = rand() < P_lick;
        Behavior.Licks(trial, t) = did_lick;
        
        % Kinematics determined by action
        if did_lick
            v_t = v_lick;
        else
            v_t = v_run;
        end
        Behavior.Velocity(trial, t) = v_t;
        dt = dx / v_t; % Time spent in current spatial bin
        
        % --- B. STATE TRANSITION (Kalman Filter) ---
        % 1. Prediction (Path integration)
        mu_pred = mu_t + dx; 
        Sigma_pred = Sigma_t + Q_noise * dx;
        
        % 2. Observation (Visual cue)
        dist_to_landmark = abs(true_x - visual_landmark);
        R_obs = R_slope * dist_to_landmark + R_min;
        
        % 3. Update
        K_t = Sigma_pred / (Sigma_pred + R_obs); % Kalman Gain
        mu_t = mu_pred + K_t * (true_x - mu_pred); % Assuming veridical obs of x
        Sigma_t = (1 - K_t) * Sigma_pred;
        
        Latents.Uncertainty(trial, t) = sqrt(Sigma_t);
        
        % Next belief state
        b_next = exp(-0.5 * ((x_space - mu_t).^2) / max(Sigma_t, 1e-3));
        b_next = b_next / sum(b_next);
        
        % --- C. REWARD & TD LEARNING ---
        if did_lick && (true_x >= reward_start) && (true_x <= reward_end) && ~reward_collected
            r_t = 1; 
            reward_collected = true;
        else
            r_t = 0;
        end
        
        % Next state value
        if t == N_bins-1
            V_next = 0; % Terminal state
        else
            V_next = sum(w_val .* b_next);
        end
        
        % Continuous TD Error (Average Reward Formulation)
        delta_t = r_t - (rho * dt) + V_next - V_t;
        Latents.RPE(trial, t) = delta_t;
        
        % Updates
        w_val = w_val + eta_w * delta_t * b_t; % Weight update spread over belief
        rho = rho + eta_rho * (r_t - rho * dt); % Update global reward rate
        
        % Step forward
        b_t = b_next;
    end
end

%% 5. Visualization 
figure('Position', [100, 100, 1200, 800], 'Name', 'Mechanistic Model Latents');

% 1. Simulated Lick Behavior
subplot(2,2,1);
imagesc(x_space, 1:N_trials, Behavior.Licks);
colormap(gca, flipud(gray));
xline(visual_landmark, 'b--', 'LineWidth', 2);
xline([reward_start, reward_end], 'g-', 'LineWidth', 2);
title('Behavior: Licking Refinement');
xlabel('Corridor Position'); ylabel('Trial');

% 2. Evolution of the Spatial Value Function V(x)
subplot(2,2,2);
imagesc(x_space, 1:N_trials, Latents.Value);
colormap(gca, parula); colorbar;
xline(visual_landmark, 'b--', 'LineWidth', 2);
xline([reward_start, reward_end], 'g-', 'LineWidth', 2);
title('Latent: State Value V(x)');
xlabel('Corridor Position'); ylabel('Trial');

% 3. Positional Uncertainty Dynamics (Kalman Convergence)
subplot(2,2,3);
hold on;
plot(x_space, Latents.Uncertainty(1, :), 'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'DisplayName', 'Trial 1');
plot(x_space, Latents.Uncertainty(50, :), 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Trial 50');
plot(x_space, Latents.Uncertainty(200, :), 'Color', [0.2 0.2 0.8], 'LineWidth', 2, 'DisplayName', 'Trial 200');
xline(visual_landmark, 'b--', 'HandleVisibility', 'off');
legend('Location', 'northwest');
title('Latent: Emergent Spatial Uncertainty (\sigma)');
xlabel('Corridor Position'); ylabel('Spatial Uncertainty');
hold off;

% 4. TD Error Map (RPE)
subplot(2,2,4);
imagesc(x_space, 1:N_trials, Latents.RPE);
colormap(gca, custom_redblue()); % Using custom colormap function below
colorbar; clim([-0.2 0.2]);
xline(visual_landmark, 'b--', 'LineWidth', 2);
xline([reward_start, reward_end], 'g-', 'LineWidth', 2);
title('Latent: Continuous TD Error (\delta)');
xlabel('Corridor Position'); ylabel('Trial');

% Inline colormap helper function
function cmap = custom_redblue()
    c1 = [0 0 1]; c2 = [1 1 1]; c3 = [1 0 0];
    m = 64;
    cmap = [linspace(c1(1),c2(1),m)', linspace(c1(2),c2(2),m)', linspace(c1(3),c2(3),m)'];
    cmap = [cmap; [linspace(c2(1),c3(1),m)', linspace(c2(2),c3(2),m)', linspace(c2(3),c3(3),m)']];
end