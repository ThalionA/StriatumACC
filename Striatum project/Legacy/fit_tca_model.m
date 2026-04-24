function P_local = fit_tca_model(data_sub, nFactors, options, tca_type)
% fit_tca_model fits a CP tensor decomposition on a numeric data array.
%
% Inputs:
%   data_sub  - numeric array (e.g., neurons x bins x trials)
%   nFactors  - number of factors for the decomposition
%   options   - structure with options for the decomposition (maxiters, tol, printitn)
%   tca_type  - string specifying the TCA method: 'cp_als', 'cp_nmu', or 'cp_orth_als'
%
% Output:
%   P_local   - fitted CP model (ktensor)
%
% This helper function creates the tensor locally on the worker to avoid
% serialization issues.

% Convert the numeric array to a tensor locally.
data_tensor = tensor(data_sub);

% Fit the TCA model based on the specified method.
switch lower(tca_type)
    case 'cp_als'
        P_local = cp_als(data_tensor, nFactors, options);
    case 'cp_nmu'
        P_local = cp_nmu(data_tensor, nFactors, options);
    case 'cp_orth_als'
        P_local = cp_orth_als(data_tensor, nFactors, options);
    otherwise
        error('Unknown TCA method');
end

end