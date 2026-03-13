%% 1. SETUP & LOAD DATA
clear; close all; clc

% Environment
env.TrackLength = 200;
env.RZ_Start    = 100;
env.RZ_End      = 125;
env.bin_size    = 2;
env.nBins       = ceil(env.TrackLength / env.bin_size);
env.x_grid      = linspace(0, env.TrackLength, env.nBins);

% --- MOCK DATA: Simulating a mouse that sharpens its map fast ---
fprintf('Generating Mock Data (Fast Map Sharpening, Slow Value Learning)...\n');
real_data = generate_sharpening_mouse(env); 

%% 2. FIT DYNAMIC MODEL
fprintf('Fitting Dual-Learning Model (Value + Map Refinement)...\n');

% Params: [Alpha, Gamma, Lambda, Cost, Sigma_Init, Sigma_Decay]
% Sigma_Final is usually fixed to a small value (e.g., 5cm) to represent asymptotic acuity
x0 = [0.1,  0.95, 0.90, 0.2,  50,  200]; 
lb = [0.01, 0.80, 0.50, 0.0,  10,  10];  
ub = [0.5,  0.99, 0.99, 1.0,  100, 1000]; 

obj_fun = @(p) get_nll_dynamic_sigma(p, real_data, env);
options = optimoptions('fmincon', 'Display', 'iter', 'DiffMinChange', 1e-3);

p_hat = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);

fitted.alpha       = p_hat(1);
fitted.gamma       = p_hat(2);
fitted.lambda      = p_hat(3);
fitted.cost        = p_hat(4);
fitted.sigma_init  = p_hat(5);
fitted.sigma_decay = p_hat(6); % Tau (in trials)

fprintf('\n--- RESULTS ---\n');
fprintf('Map starts wide (%.1f cm) and sharpens every %.0f trials.\n', ...
        fitted.sigma_init, fitted.sigma_decay);

%% 3. VISUALIZE LATENTS
visualize_dynamic_learning(real_data, fitted, env);


%% --- CORE FUNCTIONS ---

function nll = get_nll_dynamic_sigma(p_vec, data, env)
    % Unpack
    p.alpha       = p_vec(1);
    p.gamma       = p_vec(2);
    p.lambda      = p_vec(3);
    p.cost        = p_vec(4);
    p.sigma_init  = p_vec(5);
    p.sigma_tau   = p_vec(6);
    
    p.sigma_final = 5; % Assumption: Visual acuity limit
    
    nBasis = 20;
    centers = linspace(0, env.TrackLength, nBasis);
    w_critic = zeros(nBasis, 1);
    w_actor  = zeros(nBasis, 1);
    
    log_lik = 0;
    nTrials = size(data.lick_matrix, 2);
    
    for tr = 1:nTrials
        e_critic = zeros(nBasis, 1);
        e_actor  = zeros(nBasis, 1);
        
        % --- DYNAMIC SIGMA CALCULATION ---
        % Sigma decays over time (Map Refinement)
        current_sigma = p.sigma_final + (p.sigma_init - p.sigma_final) * exp(-tr / p.sigma_tau);
        
        for idx = 1:env.nBins
            pos = env.x_grid(idx);
            
            % Get features with CURRENT sharpness
            phi = get_rbf(pos, centers, current_sigma);
            
            % Policy
            logits = w_actor' * phi;
            prob_lick = 1 / (1 + exp(-logits));
            prob_lick = max(1e-4, min(0.9999, prob_lick)); 
            
            % Likelihood (Teacher Forcing)
            did_lick = data.lick_matrix(idx, tr);
            if did_lick
                log_lik = log_lik + log(prob_lick);
            else
                log_lik = log_lik + log(1 - prob_lick);
            end
            
            % RL Update (Standard Actor-Critic)
            V = w_critic' * phi;
            
            r_t = 0;
            if did_lick
                r_t = r_t - p.cost;
                if pos >= env.RZ_Start && pos <= env.RZ_End, r_t = r_t + 1; end
            end
            
            if idx == env.nBins
                V_next = 0;
            else
                phi_next = get_rbf(env.x_grid(idx+1), centers, current_sigma);
                V_next = w_critic' * phi_next;
            end
            
            delta = r_t + p.gamma * V_next - V;
            
            e_critic = p.gamma * p.lambda * e_critic + phi;
            w_critic = w_critic + p.alpha * delta * e_critic;
            
            d_log_pi = (did_lick - prob_lick) * phi;
            e_actor  = p.gamma * p.lambda * e_actor + d_log_pi;
            w_actor  = w_actor + p.alpha * delta * e_actor;
        end
    end
    nll = -log_lik;
end

function phi = get_rbf(pos, centers, width)
    phi = exp(-((pos - centers).^2) / (2 * width^2))';
    if sum(phi) > 0, phi = phi / sum(phi); end
end

function visualize_dynamic_learning(real_data, p, env)
    figure('Position', [50, 50, 1000, 600], 'Color', 'w');
    
    % 1. Recovered Behavior
    subplot(2,2,1);
    imagesc(real_data.lick_matrix);
    colormap(gca, flipud(gray));
    title('Real Behavior'); xlabel('Trial'); ylabel('Pos');
    
    % 2. Sigma Evolution
    subplot(2,2,2);
    trials = 1:size(real_data.lick_matrix,2);
    sigma_curve = 5 + (p.sigma_init - 5) * exp(-trials / p.sigma_decay);
    plot(trials, sigma_curve, 'LineWidth', 3, 'Color', 'b');
    xlabel('Trial'); ylabel('Positional Uncertainty (\sigma cm)');
    title('Map Sharpening (Latent)');
    grid on;
    
    % 3. The Effect of Sharpening (Basis Functions)
    subplot(2,2,[3 4]);
    hold on;
    x = env.x_grid;
    centers = linspace(0, 200, 20);
    c_mid = centers(10); % Plot one basis function in the middle
    
    % Early Trial
    sig_early = 5 + (p.sigma_init - 5) * exp(-1 / p.sigma_decay);
    y_early = exp(-((x - c_mid).^2) / (2 * sig_early^2));
    plot(x, y_early, 'r--', 'LineWidth', 2);
    
    % Late Trial
    sig_late = 5 + (p.sigma_init - 5) * exp(-500 / p.sigma_decay);
    y_late = exp(-((x - c_mid).^2) / (2 * sig_late^2));
    plot(x, y_late, 'g-', 'LineWidth', 2);
    
    legend('Trial 1 (Confused)', 'Trial 500 (Sharp)');
    title('How the Mouse''s Internal Map Changes');
    xlabel('Position');
end

% --- DATA GENERATOR FOR TESTING ---
function data = generate_sharpening_mouse(env)
    % A mouse that starts VERY confused (Sigma=60) but learns map fast (Tau=100)
    % But learns Value slowly (Alpha=0.05)
    p.alpha = 0.05; p.gamma = 0.95; p.lambda = 0.90; p.cost = 0.15;
    
    sigma_init = 60; sigma_final = 5; sigma_tau = 100;
    
    nTrials = 500;
    nBasis = 20; centers = linspace(0, 200, nBasis);
    w_c = zeros(nBasis,1); w_a = zeros(nBasis,1);
    lick_mat = zeros(env.nBins, nTrials);
    
    for tr = 1:nTrials
        curr_sig = sigma_final + (sigma_init - sigma_final)*exp(-tr/sigma_tau);
        e_c = zeros(nBasis,1); e_a = zeros(nBasis,1);
        
        for i = 1:env.nBins
            pos = env.x_grid(i);
            phi = get_rbf(pos, centers, curr_sig);
            
            logits = w_a' * phi;
            p_lick = 1/(1+exp(-logits));
            if rand < p_lick, lick=1; else, lick=0; end
            lick_mat(i,tr) = lick;
            
            r=0; 
            if lick
                r = r - p.cost;
                if pos>=100 && pos<=125, r=r+1; end
            end
            
            if i < env.nBins
                phi_next = get_rbf(env.x_grid(i+1), centers, curr_sig);
                V_next = w_c' * phi_next;
            else
                V_next = 0;
            end
            
            delta = r + p.gamma * V_next - (w_c'*phi);
            
            e_c = p.gamma * p.lambda * e_c + phi;
            w_c = w_c + p.alpha * delta * e_c;
            
            e_a = p.gamma * p.lambda * e_a + (lick-p_lick)*phi;
            w_a = w_a + p.alpha * delta * e_a;
        end
    end
    data.lick_matrix = lick_mat;
end