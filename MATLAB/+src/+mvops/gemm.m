function C = gemm(A, trans, B, varargin)
%% function C = gemm(A, trans, B, varargin)
% Perform matrix-matrix multiplication with a dense matrix.
% Supports different precisions for computation and output.
% Note that we should have:
%   compute_precision >= output_precision
%
% Inputs:
%   A: dense matrix
%   trans: 'N' for normal, 'T' for transpose for A only
%   B: dense matrix
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as A.
%       'precision_output': precision for output
%               Optional. Default is same as A.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   C: output matrix
%
% Example:
%   C = src.mvops.gemm(A, 'N', B, 'precision_compute', 'double', 'precision_output', 'double');
%   C = src.mvops.gemm(A, 'T', B, 'precision_compute', @single, 'precision_output', @half);

   [compute_precision, output_precision] = parse_options(A, varargin{:});
   
   A = compute_precision(A);
   B = compute_precision(B);
   
   if strcmp(trans, 'N')
      if(strcmp(char(compute_precision), 'half'))
         m = size(A, 1);
         n = size(B, 2);
         C = compute_precision(zeros(m, n));
         for i = 1:n
            C(:, i) = src.mvops.gemv(A, 'N', B(:, i), 'precision_compute', compute_precision, 'precision_output', compute_precision);
         end
      else
         C = A * B;
      end
   elseif strcmp(trans, 'T')
      if(strcmp(char(compute_precision), 'half'))
         m = size(A, 2);
         n = size(B, 2);
         C = compute_precision(zeros(m, n));
         for i = 1:n
            C(:, i) = src.mvops.gemv(A, 'T', B(:, i), 'precision_compute', compute_precision, 'precision_output', compute_precision);
         end
      else
         C = A' * B;
      end
   else
      error('Invalid trans argument');
   end
   
   C = output_precision(C);
end

function [precision_compute, precision_output] = parse_options(A, varargin)
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
   input_precision = src.utils.parse_precision(class(A));
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
      fprintf('Parameters for gemm:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end
