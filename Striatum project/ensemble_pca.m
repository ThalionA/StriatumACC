%% PCA on State-Space Trajectories by Ensemble


total_ensembles = max(ensemble_assignments);

% --- Create Figure and Tiled Layout ---
figure('Name', 'Average Trajectory by Ensemble in Separate PCA Spaces', 'Position', [50, 50, 1200, 800]);
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'normal');

% Define epochs and their corresponding colors and names from cfg
epoch_trials = {1:10, 11:20, 21:30, 31:40, 41:50};
epoch_names = [cfg.plot.epoch_names, {'Pre-Disengaged'}, {'Disengaged'}];
epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert, [0.8, 0.2, 0.2], [0.3, 0.3, 0.3]};

% --- Loop Through Each Ensemble ---
for iensemble = 1:total_ensembles
    nexttile;
    hold on;

    engaged_ensemble_tensor = cat(1, engaged_aligned_data{:, iensemble});

    disengaged_ensemble_tensor = cat(1, disengaged_aligned_data{:, iensemble});

    entire_ensemble_tensor = cat(3, engaged_ensemble_tensor, disengaged_ensemble_tensor);
    [n_neurons_ens, n_bins, n_trials] = size(entire_ensemble_tensor);


    % --- 2. Perform PCA on this Ensemble's Data ---
    % Reshape from [Neurons x Bins x Trials] to [(Bins * Trials) x Neurons]
    data_for_pca = reshape(permute(entire_ensemble_tensor, [2 3 1]), [n_bins * n_trials, n_neurons_ens]);

    % Run PCA. 'coeff' are the new axes, 'score' is the projected data.
    [coeff, score, ~, ~, explained] = pca(data_for_pca);

    % --- 3. Calculate Epoch-Averaged Trajectories in PC Space ---
    % Reshape the projected data (first 3 PCs) back to a trial structure
    score_3d = score(:, 1:3);
    trajectories_in_pc_space = reshape(score_3d, [n_bins, n_trials, 3]);

    avg_trajectories = cell(1, numel(epoch_trials));
    plot_handles = gobjects(numel(epoch_trials), 1);

    for i_epoch = 1:numel(epoch_trials)
        % Average trajectories across trials for the current epoch
        mean_traj = mean(trajectories_in_pc_space(:, epoch_trials{i_epoch}, :), 2);
        avg_trajectories{i_epoch} = squeeze(mean_traj); % Result is [Bins x 3]

        % --- 4. Plot the Trajectory for this Epoch ---
        traj = smoothdata(avg_trajectories{i_epoch}, 1, 'movmean', 10);
        plot_handles(i_epoch) = plot3(traj(:,1), traj(:,2), traj(:,3), ...
            'Color', epoch_colors{i_epoch}, 'LineWidth', 2);

        % Add markers for start (circle) and end (square) points
        scatter3(traj(1,1), traj(1,2), traj(1,3), 50, epoch_colors{i_epoch}, 'o', 'filled');
        scatter3(traj(end,1), traj(end,2), traj(end,3), 50, epoch_colors{i_epoch}, 's', 'filled');
    end

    hold off;

    % --- 5. Format Subplot ---
    title(sprintf('Ensemble %d (N=%d)', iensemble, n_neurons_ens));
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
    zlabel(sprintf('PC3 (%.1f%%)', explained(3)));
    grid on;
    axis tight;
    view(35, 25); % Set consistent viewing angle
end

% --- Add Shared Legend to the Figure ---
lg = legend(plot_handles, epoch_names, 'FontSize', 10);
lg.Layout.Tile = 'east'; % Place legend in its own tile on the side


%% Subspace and Vector Angle Analysis (Corrected)
total_ensembles = max(ensemble_assignments);

% Pre-allocate structure for results
results = struct();

fprintf('\n--- Subspace Geometry Analysis ---\n');
fprintf('%-10s | %-12s | %-12s | %-12s\n', 'Ensemble', 'Vec Angle', 'Plane Ang 1', 'Plane Ang 2');
fprintf('--------------------------------------------------------------\n');

for iens = 1:total_ensembles
    % --- 1. Data Preparation ---
    eng_data = cat(1, engaged_aligned_data{:, iens}); 
    dis_data = cat(1, disengaged_aligned_data{:, iens});
    
    full_tensor = cat(3, eng_data, dis_data);
    [n_neurons, n_bins, n_total_trials] = size(full_tensor);
    
    % Define Epoch Indices
    idx_naive  = 1:10;
    idx_expert = 21:30;
    idx_dis    = 41:50; 
    
    % Helper to flatten data
    get_flat = @(idx) reshape(permute(full_tensor(:,:,idx), [2 3 1]), [], n_neurons);
    
    X_naive  = get_flat(idx_naive);
    X_expert = get_flat(idx_expert);
    X_dis    = get_flat(idx_dis);
    
    % --- 2. Vector Angle Analysis (Centroid Shifts) ---
    % FIX 1: Use 'omitnan' to prevent NaN propagation
    mu_naive  = mean(X_naive, 1, 'omitnan');   % 1 x N
    mu_expert = mean(X_expert, 1, 'omitnan');  % 1 x N
    mu_dis    = mean(X_dis, 1, 'omitnan');     % 1 x N
    
    % Define Vectors
    v_learn = mu_expert - mu_naive;
    v_dis   = mu_dis - mu_expert;
    
    % Calculate Angle
    % Handle potential zero-norm vectors (rare, but good safety)
    if norm(v_learn) > 0 && norm(v_dis) > 0
        cos_theta = dot(v_learn, v_dis) / (norm(v_learn) * norm(v_dis));
        vec_angle_deg = rad2deg(acos(cos_theta));
    else
        vec_angle_deg = NaN; 
    end
    
    % --- 3. Principal Angles (Plane to Plane) ---
    % Handle NaNs in PCA data if necessary
    try
        [coeff_exp, ~, ~] = pca(X_expert, 'Rows', 'complete');
        [coeff_dis, ~, ~] = pca(X_dis, 'Rows', 'complete');
        
        if n_neurons >= 2 && size(coeff_exp,2)>=2 && size(coeff_dis,2)>=2
            Q_expert = coeff_exp(:, 1:2); 
            Q_dis    = coeff_dis(:, 1:2); 
            
            S = svd(Q_expert' * Q_dis);
            plane_angles_deg = rad2deg(acos(S));
        else
            plane_angles_deg = [NaN; NaN];
        end
    catch
        plane_angles_deg = [NaN; NaN];
    end
    
    % Store Results
    results(iens).vec_angle = vec_angle_deg;
    results(iens).plane_angles = plane_angles_deg;
    results(iens).v_learn = v_learn; % 1 x N
    results(iens).v_dis = v_dis;     % 1 x N
    results(iens).centroids = [mu_naive; mu_expert; mu_dis]; % 3 x N matrix
    
    fprintf('Ens %-6d | %-12.1f | %-12.1f | %-12.1f\n', ...
        iens, vec_angle_deg, plane_angles_deg(1), plane_angles_deg(2));
end

%% --- Visualization (Corrected Matrix Dimensions) ---

target_ens = 5; % Ens 4 shows clear structure
vL = results(target_ens).v_learn; % 1 x N
vD = results(target_ens).v_dis;   % 1 x N
centroids = results(target_ens).centroids; % 3 x N

% FIX 2: Check for NaNs before plotting
if any(isnan(vL)) || any(isnan(vD))
    warning('Selected ensemble has NaN vectors. Cannot plot.');
else
    % Create orthonormal basis
    % Note: Transpose basis vectors to columns for the projection matrix
    basis1 = vL / norm(vL); % 1 x N
    
    resid = vD - (dot(vD, basis1) * basis1);
    basis2 = resid / norm(resid); % 1 x N
    
    % FIX 3: Transpose bases to make [N x 2] matrix
    projection_matrix = [basis1', basis2']; 
    
    % Project the centroids: [3 x N] * [N x 2] = [3 x 2]
    p_centroids = centroids * projection_matrix; 
    
    figure('Name', sprintf('Geometry of State Change: Ensemble %d', target_ens), 'Color', 'w');
    hold on;
    
    % Plot the path
    plot(p_centroids(:,1), p_centroids(:,2), '--k', 'LineWidth', 1);
    
    % Plot points
    scatter(p_centroids(1,1), p_centroids(1,2), 100, [0 0.447 0.741], 'filled', 'DisplayName', 'Naive'); 
    scatter(p_centroids(2,1), p_centroids(2,2), 100, [0.466 0.674 0.188], 'filled', 'DisplayName', 'Expert'); 
    scatter(p_centroids(3,1), p_centroids(3,2), 100, [0.2 0.2 0.2], 'filled', 'DisplayName', 'Disengaged'); 
    
    % Quivers
    quiver(p_centroids(1,1), p_centroids(1,2), ...
           p_centroids(2,1)-p_centroids(1,1), p_centroids(2,2)-p_centroids(1,2), ...
           0, 'Color', [0.466 0.674 0.188], 'LineWidth', 2, 'MaxHeadSize', 0.5, 'HandleVisibility','off');
    
    quiver(p_centroids(2,1), p_centroids(2,2), ...
           p_centroids(3,1)-p_centroids(2,1), p_centroids(3,2)-p_centroids(2,2), ...
           0, 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'HandleVisibility','off');
    
    xlabel('Learning Axis (Normalized)');
    ylabel('Orthogonal Axis');
    title(sprintf('Ensemble %d: Orthogonality of Disengagement\nVector Angle: %.1f^{\\circ}', ...
        target_ens, results(target_ens).vec_angle));
    grid on; axis equal;
    legend('Path', 'Naive', 'Expert', 'Disengaged');
    hold off;
end

%% Trajectory Dynamics Analysis: Velocity Field Similarity (All Epochs)
% Computes the similarity of flow fields (diffs) across trials and 4 epochs.

% --- Configuration ---
total_ensembles = max(ensemble_assignments);

% Epoch Definitions:
% 1. Naive (1-10)
% 2. Expert (21-30) -> skipping 'Intermediate' (11-20) to focus on stable states
% 3. Pre-Disengaged (31-40) -> The transition zone
% 4. Disengaged (41-50)
epochs = {1:10, 21:30, 31:40, 41:50}; 
epoch_names = {'Naive', 'Expert', 'Pre-Dis', 'Disengaged'};
n_epochs = length(epochs);

% Smoothing window (ensure this isn't too large relative to bin count)
smooth_window = 10; 

figure('Name', 'Dynamics Similarity: With Pre-Disengaged', 'Position', [50, 50, 1400, 700]);
t = tiledlayout(2, total_ensembles, 'TileSpacing', 'compact');

% Store results for text output
summary_stats = struct();

for iens = 1:total_ensembles
    
    % --- 1. Data Tensor Construction ---
    eng_data = cat(1, engaged_aligned_data{:, iens}); 
    dis_data = cat(1, disengaged_aligned_data{:, iens});
    full_tensor = cat(3, eng_data, dis_data); % [Neurons x Bins x Trials]
    
    [n_neurons, n_bins, n_total_trials] = size(full_tensor);
    
    % Pre-allocate
    mean_velocities = {}; 
    consistency_scores = cell(1, n_epochs); 
    
    % --- 2. Process Each Epoch ---
    for iepoch = 1:n_epochs
        trial_idxs = epochs{iepoch};
        n_trials = length(trial_idxs);
        epoch_data = full_tensor(:, :, trial_idxs);
        
        % A. Compute Individual Trial Velocities
        trial_vels_flat = [];
        
        for ti = 1:n_trials
            raw_traj = epoch_data(:, :, ti);
            
            % Smooth 
            traj_smooth = smoothdata(raw_traj, 2, 'movmean', smooth_window, 'omitnan');
            
            % 1st Derivative (Velocity): [Neurons x Bins-1]
            vel = diff(traj_smooth, 1, 2);
            
            % Flatten to column for correlation
            trial_vels_flat = [trial_vels_flat, vel(:)]; 
        end
        
        % B. Compute Mean Trajectory & Its Velocity (The "Canonical" Flow)
        mean_traj = mean(epoch_data, 3, 'omitnan');
        mean_traj_smooth = smoothdata(mean_traj, 2, 'movmean', smooth_window, 'omitnan');
        mean_vel = diff(mean_traj_smooth, 1, 2);
        
        % Store for Between-Epoch Comparison
        mean_velocities{iepoch} = mean_vel(:);
        
        % --- Analysis 1: Trial-to-Mean Consistency ---
        % Correlate each trial's velocity flow with the mean flow of that epoch
        curr_consistency = zeros(n_trials, 1);
        v_mean_flat = mean_vel(:);
        
        for ti = 1:n_trials
            v_trial = trial_vels_flat(:, ti);
            if norm(v_trial)>0 && norm(v_mean_flat)>0
                curr_consistency(ti) = dot(v_trial, v_mean_flat) / (norm(v_trial) * norm(v_mean_flat));
            else
                curr_consistency(ti) = NaN;
            end
        end
        consistency_scores{iepoch} = curr_consistency;
    end
    
    % --- Analysis 2: Between-Epoch Evolution Matrix ---
    vel_sim_matrix = zeros(n_epochs, n_epochs);
    for i = 1:n_epochs
        for j = 1:n_epochs
            v1 = mean_velocities{i};
            v2 = mean_velocities{j};
            
            if norm(v1)>0 && norm(v2)>0
                vel_sim_matrix(i,j) = dot(v1, v2) / (norm(v1) * norm(v2));
            end
        end
    end
    
    % --- Visualization ---
    
    % Row 1: Consistency (Boxplot)
    nexttile(iens);
    % Prepare data for boxplot
    plot_data = [];
    plot_grps = [];
    for ie = 1:n_epochs
        d = consistency_scores{ie};
        plot_data = [plot_data; d];
        plot_grps = [plot_grps; repmat(ie, length(d), 1)];
    end
    
    boxplot(plot_data, plot_grps, 'Labels', epoch_names);
    title(sprintf('Ens %d: Consistency', iens));
    if iens==1; ylabel('Cos Sim (Trial vs Mean)'); end
    ylim([0, 1]); 
    grid on;
    
    % Row 2: Evolution (Heatmap)
    nexttile(total_ensembles + iens);
    imagesc(vel_sim_matrix);
    colormap('parula'); clim([0, 1]);
    title(sprintf('Ens %d: Similarity', iens));
    xticks(1:n_epochs); xticklabels(epoch_names);
    yticks(1:n_epochs); yticklabels(epoch_names);
    xtickangle(45);
    
    summary_stats(iens).vel_matrix = vel_sim_matrix;
end

cb = colorbar; 
cb.Layout.Tile = 'east';

sgtitle('Dynamics Analysis: Velocity Field Evolution');

% Text Summary
fprintf('\n--- Dynamical Shape Similarity (Velocity Field) ---\n');
% Comparison columns: 
% 1. Naive -> Expert (Learning)
% 2. Expert -> Pre-Dis (Stability Check)
% 3. Pre-Dis -> Disengaged (The "Drop")
% 4. Expert -> Disengaged (Total Shift)

fprintf('%-8s | %-10s | %-10s | %-10s | %-10s\n', 'Ens', 'Nav-Exp', 'Exp-Pre', 'Pre-Dis', 'Exp-Dis');
fprintf('------------------------------------------------------------------\n');

for iens = 1:total_ensembles
    M = summary_stats(iens).vel_matrix;
    % Indices: 1=Naive, 2=Expert, 3=Pre-Dis, 4=Disengaged
    
    nav_exp = M(1,2);
    exp_pre = M(2,3);
    pre_dis = M(3,4);
    exp_dis = M(2,4);
    
    fprintf('Ens %-4d | %-10.3f | %-10.3f | %-10.3f | %-10.3f\n', ...
        iens, nav_exp, exp_pre, pre_dis, exp_dis);
end