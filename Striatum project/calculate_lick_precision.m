function p = calculate_lick_precision(l, rz_start)

l(l>rz_start) = nan;
p = sum((l-rz_start).^2, 'omitmissing');
