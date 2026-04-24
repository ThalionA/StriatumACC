function results = decode_ensemble_ablation(supermouse_tensor, ensemble_assignments, cfg)
%DECODE_ENSEMBLE_ABLATION  Cross‑validated spatial decoding with ensemble knock‑outs and single‑ensemble decoders.
%
%  results = decode_ensemble_ablation(tensor, ens_idx, cfg)
%
%  This version (2025‑05‑22) fixes the SHUFFLE‑CONTROL bug: shuffled datasets
%  now randomise the neuron–position correspondence **independently for every
%  trial**, yielding genuine chance‑level performance.
%
%  Implements three decoders:
%        • Baseline  – all neurons.
%        • Knock‑out – remove ensemble *i* (results.knockout(i)).
%        • Only      – keep ensemble *i* alone  (results.only(i)).
%
%  INPUTS
%        supermouse_tensor      (neurons × bins × trials) rate tensor.
%        ensemble_assignments   (#neurons × 1) ensemble id 1…N.
%        cfg (struct, optional)
%             .cv_type        'leave-one-out' | 'kfold'  [ 'leave-one-out' ]
%             .kfolds         folds if cv_type=='kfold'  [ 5 ]
%             .shuffle_iters  control iterations         [ 50 ]
%             .poisson_eps    min rate for log()         [ 1e-6 ]
%             .do_knockout    run KO decoders            [ true ]
%             .do_only        run single‑ensemble dec.   [ true ]
%             .verbose        talk to screen             [ true ]
%
%  OUTPUT  (results struct)
%        .baseline           – decoding summary
%        .shuffle            – fields: abs_error_space_mean/sem, …
%        .knockout(i)        – summary (if do_knockout)
%        .only(i)            – summary (if do_only)
%
%  EXAMPLE
%        res = decode_ensemble_ablation(supermouse_valid, ens_idx);
%


arguments
    supermouse_tensor      double
    ensemble_assignments   double
    cfg.struct = struct()
end

%% === Defaults ===
D = struct('cv_type','leave-one-out', 'kfolds',5, 'shuffle_iters',50, ...
           'poisson_eps',1e-6, 'do_knockout',true, 'do_only',true, ...
           'verbose',true);
cfg = filldefaults(cfg, D);

[nNeurons,nBins,nTrials] = size(supermouse_tensor);
assert(numel(ensemble_assignments)==nNeurons, 'ensemble_assignments size mismatch');
ensembles = unique(ensemble_assignments(:)');

%% === Helper handles ===
poisLL = @(r,lam) r.*log(max(lam,cfg.poisson_eps)) - lam;   % elementwise

    function pred = decode_once(tensor, mask)
        idx  = find(mask);
        pred = nan(nTrials,nBins);
        for tst = 1:nTrials
            trn = setdiff(1:nTrials,tst);
            % mu  = mean(tensor(idx,:,trn),3,'omitnan');        % (#idx × bins)
            mu  = mean(tensor(idx,:,21:30),3,'omitnan');        % (#idx × bins)
            testDat = tensor(idx,:,tst);
            for b = 1:nBins
                ll = sum(poisLL(testDat(:,b), mu),1);        % 1×bins
                [~,pred(tst,b)] = max(ll);
            end
        end
    end

    function S = summarise(pred)
        trueMat = repmat(1:nBins, nTrials,1);
        err     = pred - trueMat;
        S.abs_error_space = mean(abs(err),1,'omitnan');  % 1×bins
        S.abs_error_trial = mean(abs(err),2,'omitnan');  % trials×1
        S.rmse            = sqrt(mean(err(:).^2,'omitnan'));
    end

%% === Baseline ===
if cfg.verbose, fprintf('Baseline decoding with all %d neurons…\n',nNeurons); end
results.baseline = summarise(decode_once(supermouse_tensor, true(nNeurons,1)));

%% === Shuffle control  (chance‑level expectation) ===
if cfg.shuffle_iters>0
    if cfg.verbose, fprintf('Running %d shuffled controls…\n',cfg.shuffle_iters); end
    shuffle_err = nan(cfg.shuffle_iters,nTrials,nBins);
    for s = 1:cfg.shuffle_iters
        tmp = supermouse_tensor;
        %  ❱❱ Randomise position labels independently per trial and neuron
        for n = 1:nNeurons
            for t = 1:nTrials
                tmp(n,:,t) = tmp(n, randperm(nBins), t);
            end
        end
        %  Decode
        predS = decode_once(tmp, true(nNeurons,1));
        shuffle_err(s,:,:) = predS - repmat(1:nBins,nTrials,1);
    end
    results.shuffle.abs_error_space_mean = squeeze(mean(mean(abs(shuffle_err),2),1));
    results.shuffle.abs_error_space_sem  = squeeze(std(mean(abs(shuffle_err),2),0,1))/sqrt(cfg.shuffle_iters);
    results.shuffle.abs_error_trial_mean = squeeze(mean(mean(abs(shuffle_err),3),1));
    results.shuffle.abs_error_trial_sem  = squeeze(std(mean(abs(shuffle_err),3),0,1))/sqrt(cfg.shuffle_iters);
    results.shuffle.rmse_mean = mean(sqrt(mean(shuffle_err.^2,[2 3])));
    results.shuffle.rmse_sem  = std(sqrt(mean(shuffle_err.^2,[2 3])))/sqrt(cfg.shuffle_iters);
else
    results.shuffle = struct();
end

%% === Ensemble knock‑out decoders ===
if cfg.do_knockout
    if cfg.verbose, fprintf('Running ensemble knock‑outs…\n'); end
    for e = ensembles
        mask = ensemble_assignments ~= e;
        results.knockout(e) = summarise(decode_once(supermouse_tensor, mask)); %#ok<AGROW>
    end
end

%% === Single‑ensemble decoders ===
if cfg.do_only
    if cfg.verbose, fprintf('Running single‑ensemble decoders…\n'); end
    for e = ensembles
        mask = ensemble_assignments == e;
        results.only(e) = summarise(decode_once(supermouse_tensor, mask)); %#ok<AGROW>
    end
end

end  % main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cfg = filldefaults(cfg, def)
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i}))
        cfg.(fn{i}) = def.(fn{i});
    end
end
end
