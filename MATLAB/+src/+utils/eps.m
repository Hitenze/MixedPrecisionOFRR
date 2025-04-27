function val = eps(precision)
%% function val = eps(precision)
% Get the machine epsilon of a precision (function handle or string)
%
% Inputs:
%   precision: precision type as a function handle (@double, @single, @half)
%              or as a string ('double', 'single', 'half')
%
% Outputs:
%   val: machine epsilon for the specified precision
%
% Example:
%   eps_double = src.utils.eps(@double);
%   eps_single = src.utils.eps('single');
%   eps_half = src.utils.eps('half');

   precision = src.utils.parse_precision(precision);
   
   if isequal(precision, @double)
      val = eps('double');
   elseif isequal(precision, @single)
      val = eps('single');
   elseif isequal(precision, @half)
      val = half(2^(-10));
   else
      warning('Unknown precision type. Using double precision epsilon.');
      val = eps('double');
   end
end
