function p = scalar_to_quantile(data, x)
    % Ensure data is sorted
    sorted_data = sort(data);
    
    % Find the number of elements in the data
    n = length(sorted_data);
    
    % Find the rank of x in the sorted data
    rank = sum(sorted_data <= x);
    
    % Calculate the percentile (rank/n * 100)
    p = (rank / n);
end