function g = gini_coeff(x)

sorted_x = sort(x);

g = sum((2*(1:numel(x)) - numel(x) - 1).*sorted_x)/((numel(x)^2)*mean(x));

end