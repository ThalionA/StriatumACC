function p = calculate_lick_precision(l, rz_start)

p = sum((l-rz_start).^2);
