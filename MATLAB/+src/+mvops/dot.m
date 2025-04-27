function y = dot(a, b, varargin)
%% function y = dot(a, b, varargin)
% Compute the dot product of two vectors with precision control.
% Supports different precisions for computation and output.
% Note that we should have:
%   precision_compute >= precision_output
%
% Inputs:
%   a: first vector
%   b: second vector
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as a.
%       'precision_output': precision for output
%               Optional. Default is same as a.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   y: dot product result
%
% Example:
%   y = src.mvops.dot(a, b, 'precision_compute', 'double', 'precision_output', 'double');
%   y = src.mvops.dot(a, b, 'precision_compute', @single, 'precision_output', @half);

   assert(strcmp(class(a), class(b)), 'Vectors a and b must have the same precision in dot product');
   assert(length(a) == length(b), 'Vectors a and b must have the same length');
   
   [precision_compute, precision_output] = parse_options(a, varargin{:});
   
   a = precision_compute(a);
   b = precision_compute(b);
   
   if(strcmp(char(precision_compute), 'half'))
      y = sum(a .* b);
   else
      y = a' * b;
   end
   
   y = precision_output(y);
end

function [precision_compute, precision_output] = parse_options(a, varargin)
   if nargin > 2
      for i = 1:2:nargin-1
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'check_params'
               check_params = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   input_precision = src.utils.parse_precision(class(a));
   if ~exist('precision_compute', 'var')
      precision_compute = input_precision;
   end
   if ~exist('precision_output', 'var')
      precision_output = input_precision;
   end

   if(strcmp(char(precision_output), 'double'))
      assert(strcmp(char(precision_compute), 'double'), 'precision_compute must be double if precision_output is double');
   elseif(strcmp(char(precision_output), 'single'))
      assert(strcmp(char(precision_compute), 'single') || strcmp(char(precision_compute), 'double'), ... 
             'precision_compute must be single or double if precision_output is single');
   end

   if ~exist('check_params', 'var')
      check_params = false;
   end
   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for dot:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end 