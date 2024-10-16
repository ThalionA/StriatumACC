function [p, shuffled_lick_precision_mean, shuffled_lick_precision_std, chance_quantile] = calculate_lick_precision(l, rz_start)

l(l>rz_start) = nan;
p = sum((l-rz_start).^2, 2, 'omitmissing');

lick_number = sum(~isnan(l));

shuffled_lick_precision_mean = nan;
shuffled_lick_precision_std = nan;
chance_quantile = nan;

if lick_number > 0
    shuffled_l = rz_start.*rand(500, lick_number);
    shuffled_lick_precision = sum((shuffled_l-rz_start).^2, 2);
    shuffled_lick_precision_mean = mean(shuffled_lick_precision);
    shuffled_lick_precision_std = std(shuffled_lick_precision);
    chance_quantile = scalar_to_quantile(shuffled_lick_precision, p);
end