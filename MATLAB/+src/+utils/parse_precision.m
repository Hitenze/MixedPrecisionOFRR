function precision = parse_precision(precision)
%% function precision = parse_precision(precision)
% Convert string precision to function handle if needed
%
% Inputs:
%   precision: precision string or function handle
%
% Outputs:
%   precision: function handle
%
% Example:
%   precision_fn = src.utils.parse_precision('double');
%   precision_fn = src.utils.parse_precision(@single);

   if ischar(precision) || isstring(precision)
      precision = str2func(precision);
   end
end