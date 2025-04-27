function norm2 = norm2(x, varargin)
%% function norm2 = norm2(x, varargin)
% A stable version of sqrt(x'*x)
%
% Inputs:
%   x: vector
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as x.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%       'tol': tolerance for treating values as zero
%               Optional. Default is machine epsilon.
%
% Outputs:
%   norm2: norm of x
%
% Example:
%   x = [1e10; 1e-10];
%   n = src.mvops.norm2(x);  % Stable computation to avoid overflow
%   n = src.mvops.norm2(x, 'precision_compute', 'double', 'precision_output', 'single');
%   n = src.mvops.norm2(x, 'tol', 1e-6);  % Use custom tolerance

   [precision_compute, precision_output, tol] = parse_options(x, varargin{:});
   
   % Compute in compute precision
   maxx = max(abs(precision_compute(x)));
   if maxx < tol
      % In this function we just need to make sure that we do not divide by a too small number
      norm2 = sqrt(sum(abs(precision_compute(x)).^2));
   else
      norm2 = precision_output(maxx * sqrt(sum(abs(precision_compute(x)/maxx).^2)));
   end
end

function [precision_compute, precision_output, tol] = parse_options(x, varargin)
   if length(varargin) >= 2
      for i = 1:2:length(varargin)
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'tol'
               tol = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   
   if ~exist('precision_compute', 'var')
      precision_compute = src.utils.parse_precision(class(x));
   end
   if ~exist('precision_output', 'var')
      precision_output = precision_compute;
   end
   if ~exist('tol', 'var')
      tol = src.utils.eps(char(precision_compute));
   end
end