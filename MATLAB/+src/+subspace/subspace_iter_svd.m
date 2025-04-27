function [U, S, V] = subspace_iter_svd(A, m, n, k, varargin)
%% function [U, S, V] = subspace_iter_svd(A, m, n, k, varargin)
% Subspace iteration method for computing singular value decomposition
%
% Inputs:
%   A: matrix to perform subspace iteration on
%   m: number of rows of A
%   n: number of columns of A
%   k: subspace dimension
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as A.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%       'seed': seed for random initialization
%               Optional. Default is 42.
%       'maxiter': maximum number of iterations
%               Optional. Default is 5.
%       'step_size': step size for the subspace iteration
%               Optional. Default is 1.
%       'function_orth': function for orthogonalization
%               Optional. Should accept a matrix and return orthogonalized version.
%               In the form [Q, R] = function_orth(V), where V is a matrix and Q is orthogonal.
%               Default creates a wrapper around src.qr.mgs with the specified precision.
%               Custom functions should handle precision internally.
%       'function_rr': function for Rayleigh-Ritz projection
%               Optional. Should accept a matrix and two orthogonal bases.
%               In the form [U, S, V] = function_rr(A, P, Q), where P and Q are orthogonal.
%               Default creates a wrapper around src.rr.svd_rr with the specified precision.
%               Custom functions should handle precision internally.
%       'matvec': function for matrix-vector product
%               Optional. Default is @gemm.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   U: left singular vectors
%   S: diagonal matrix of singular values
%   V: right singular vectors
%
% Example:
%   [U, S, V] = src.subspace.subspace_iter_svd(A, m, n, 10, 'precision_compute', 'double', 'precision_output', 'single');
%   [U, S, V] = src.subspace.subspace_iter_svd(A, m, n, 5, 'maxiter', 10, 'step_size', 2);

   [precision_compute, precision_output, seed, maxiter, step_size, function_orth, function_rr, matvec, check_params] = parse_options(varargin{:});
   if check_params
      U = [];
      S = [];
      V = [];
      return;
   end
   
   k = min(k, n);

   % Initialize with random matrix in output precision
   rng(seed);
   V = precision_output(randn(n, k));
   V = src.mvops.scalecols(V, 'precision_compute', precision_compute, 'precision_output', precision_output);
   if ~isstruct(A)
      A = precision_output(A);
   end

   for i = 1:maxiter
      for j = 1:step_size
         U = matvec(A, 'N', V, 'precision_compute', precision_compute, 'precision_output', precision_output);
         U = src.mvops.scalecols(U, 'precision_compute', precision_compute, 'precision_output', precision_output);
         V = matvec(A, 'T', U, 'precision_compute', precision_compute, 'precision_output', precision_output);
         V = src.mvops.scalecols(V, 'precision_compute', precision_compute, 'precision_output', precision_output);
      end
      
      % Orthogonalize and perform SVD via Rayleigh-Ritz
      [P, ~] = function_orth(U);
      [Q, ~] = function_orth(V);
      [U, S, V] = function_rr(A, P, Q);
      U = precision_output(U);
      V = precision_output(V);
   end
end

function [precision_compute, precision_output, seed, maxiter, step_size, function_orth, function_rr, matvec, check_params] = parse_options(varargin)
   if nargin > 1
      for i = 1:2:nargin
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'seed'
               seed = varargin{i+1};
            case 'maxiter'
               maxiter = varargin{i+1};
            case 'step_size'
               step_size = varargin{i+1};
            case 'function_orth'
               function_orth = varargin{i+1};
            case 'function_rr'   
               function_rr = varargin{i+1};
            case 'matvec'
               matvec = varargin{i+1};
            case 'check_params'
               check_params = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   
   if ~exist('precision_compute', 'var')
      precision_compute = @double;
   end
   if ~exist('precision_output', 'var')
      precision_output = precision_compute;
   end
   if ~exist('seed', 'var')
      seed = 42;
   end
   if ~exist('maxiter', 'var')
      maxiter = 5;
   end
   if ~exist('step_size', 'var')
      step_size = 1;
   end
   if ~exist('function_orth', 'var')
      function_orth = @(V) src.qr.mgs(V, 'precision_compute', precision_compute, 'precision_output', precision_output);
   end
   if ~exist('function_rr', 'var')
      function_rr = @(A, P, Q) src.rr.svd_rr(A, P, Q, 'precision_compute', precision_compute, 'precision_output', precision_output);
   end
   if ~exist('matvec', 'var')
      matvec = @src.mvops.gemm;
   end
   if ~exist('check_params', 'var')
      check_params = false;
   end

   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for subspace iteration:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- seed: %d\n', seed);
      fprintf(' -- maxiter: %d\n', maxiter);
      fprintf(' -- step_size: %d\n', step_size);
      fprintf(' -- function_orth: %s\n', char(function_orth));
      fprintf(' -- function_rr: %s\n', char(function_rr));
      fprintf(' -- matvec: %s\n', char(matvec));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end