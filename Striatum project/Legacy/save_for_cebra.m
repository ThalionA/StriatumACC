for ianimal = 1:n_animals

    n_trials = preprocessed_data(ianimal).n_trials;

    lick_data = preprocessed_data(ianimal).spatial_binned_data.licks(1:n_trials, :)';
    lick_data(lick_data > quantile(lick_data, 0.99, "all")) = nan;
    neural_data = preprocessed_data(ianimal).spatial_binned_fr_all;

    lick_errors = preprocessed_data(ianimal).zscored_lick_errors;

    filename = sprintf('cebra_mouse%ddata', ianimal);
    save(filename, 'neural_data', 'lick_data', 'lick_errors');

end