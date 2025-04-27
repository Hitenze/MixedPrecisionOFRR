function [U] = scalecols(U, varargin)
%% function U = scalecols(U, varargin)
% Scale columns of U to have unit infinity norm
%
% Inputs:
%   U: matrix to scale
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as U.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%
% Outputs:
%   U: scaled matrix
%
% Example:
%   U = rand(10, 5);
%   U_scaled = src.mvops.scalecols(U, 'precision_compute', 'double', 'precision_output', 'single');
%   U_scaled = src.mvops.scalecols(U); % Use same precision as input

   [precision_compute, precision_output] = parse_options(U, varargin{:});
   
   % Initialize in output precision
   U = precision_output(U);

   [~,n] = size(U);
   for j=1:n
      [~,k] = max(abs(U(:,j)));
      t = precision_compute(U(k,j));
      if (t ~= 0.0)
         U(:,j) = precision_output(precision_compute(U(:,j)) / t);
      end
   end
end

function [precision_compute, precision_output] = parse_options(U, varargin)
   if length(varargin) >= 2
      for i = 1:2:length(varargin)
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   
   if ~exist('precision_compute', 'var')
      precision_compute = src.utils.parse_precision(class(U));
   end
   if ~exist('precision_output', 'var')
      precision_output = precision_compute;
   end
end