function y = gemv(A, trans, x, varargin)
%% function y = gemv(A, trans, x, varargin)
% Perform matrix-vector multiplication with a dense matrix.
% Supports different precisions for computation and output.
% Note that we should have:
%   compute_precision >= output_precision
%
% Inputs:
%   A: dense matrix
%   trans: 'N' for normal, 'T' for transpose
%   x: input vector
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as x.
%       'precision_output': precision for output
%               Optional. Default is same as x.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   y: output vector
%
% Example:
%   y = src.mvops.gemv(A, 'N', x, 'precision_compute', 'double', 'precision_output', 'double');
%   y = src.mvops.gemv(A, 'T', x, 'precision_compute', @single, 'precision_output', @half);

   [compute_precision, output_precision] = parse_options(x, varargin{:});
   
   A = compute_precision(A);
   x = compute_precision(x);
   
   if strcmp(trans, 'N')
      if(strcmp(char(compute_precision), 'half'))
         [m, n] = size(A);
         if(n >= 1)
            y = A(:, 1) * x(1);
            for i = 2:n
               y = y + A(:, i) * x(i);
            end
         else
            y = compute_precision(zeros(m, 1));
         end
      else
         y = A * x;
      end
   elseif strcmp(trans, 'T')
      if(strcmp(char(compute_precision), 'half'))
         [m, n] = size(A);
         if(m >= 1)
            y = A(1, :)' * x(1);
            for i = 2:m
               y = y + A(i, :)' * x(i);
            end
         else
            y = compute_precision(zeros(n, 1));
         end
      else
         y = A' * x;
      end
   else
      error('Invalid trans argument');
   end
   
   y = output_precision(y);
end

function [precision_compute, precision_output] = parse_options(x, varargin)
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
   input_precision = src.utils.parse_precision(class(x));
   if ~exist('precision_compute', 'var')
      precision_compute = input_precision;
   end
   if ~exist('precision_output', 'var')
      precision_output = input_precision;
   end

   if(strcmp(char(precision_output), 'double'))
      assert(strcmp(char(precision_compute), 'double'), 'compute_precision must be double if output_precision is double');
   elseif(strcmp(char(precision_output), 'single'))
      assert(strcmp(char(precision_compute), 'single') || strcmp(char(precision_compute), 'double'), ... 
             'compute_precision must be single or double if output_precision is single');
   end

   if ~exist('check_params', 'var')
      check_params = false;
   end
   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for gemv:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end
