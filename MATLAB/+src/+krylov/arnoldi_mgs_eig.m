function [V, D] =  arnoldi_mgs_eig(A, n, k, varargin)
%% function [V, D] = arnoldi_mgs_eig(A, n, k, varargin)
% Subspace iteration method for computing eigenvalues and eigenvectors
%
% Inputs:
%   A: matrix to perform subspace iteration on
%   n: number of rows of A (A should be square matrix)
%   k: number of eigenvalues to compute
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as A.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%       'precision_eigenvalues': precision for the final eigenvalues problem
%               Optional. Default is double.
%       'precision_A_matvec_compute': precision of the matrix-vector multiplication for A
%               Optional. Default is same as precision_compute.
%       'precision_A_matvec_output': precision of the matrix-vector multiplication for A
%               Optional. Default is same as precision_A_matvec_compute.
%       'seed': seed for random initialization
%               Optional. Default is 42.
%       'maxiter': maximum number of iterations
%               Optional. Default is 5.
%       'kdim': Krylov subspace dimension for the Arnoldi process
%               Optional. Default is maxiter.
%       'reorth': boolean to indicate if reorthogonalization is used
%               Optional. Default is true.
%       'generalized': boolean to indicate if generalized eigenvalue problem is solved
%               Optional. Default is false.
%       'generalized_tol': tolerance for generalized eigenvalue problem
%               Optional. Default is 0.0.
%       'select_tol': tolerance for stop the iteration
%               Optional. Default is machine epsilon.
%       'matvec': function for matrix-vector product
%               Optional. Default is @gemm.
%       'check_params': only parse parameters and display
%
% Outputs:
%   V: approximate eigenvectors
%   D: diagonal matrix of approximate eigenvalues
%
% Example:
%   [V, D] = src.subspace.subspace_iter_eig(A, size(A, 1), 10, 'precision_compute', 'double', 'precision_output', 'single');
%   [V, D] = src.subspace.subspace_iter_eig(A, size(A, 1), 10, 'maxiter', 10, 'step_size', 2);

   [precision_compute, precision_output, precision_eigenvalues, precision_A_matvec_compute, precision_A_matvec_output, seed, maxiter, kdim, reorth, generalized, generalized_tol, select_tol, matvec, check_params] = parse_options(varargin{:});

   if check_params
      V = [];
      D = [];
      return;
   end

   k = min(k, n);

   % Initialize placeholders
   V = precision_output(zeros(n, kdim+1));

   % Initialize with random vector in output precision
   rng(seed);
   V(:,1) = precision_output(randn(n, 1));

   % Start the hessenberg process, first find the largest entry and scale the current columns
   tits = 1;
   while tits < maxiter
      % main outer loop
      % normalize the first column
      t = src.mvops.norm2(V(:,1), 'precision_compute', precision_compute, 'precision_output', precision_output);
      if t < select_tol
         break;
      else
         V(:,1) = precision_output(precision_compute(V(:,1)) / precision_compute(t));
      end
      for i = 1:kdim
         i1 = i + 1;
         V(:,i1) = precision_output(matvec(A, 'N', precision_A_matvec_compute(V(:,i)), 'precision_compute', precision_A_matvec_compute, 'precision_output', precision_A_matvec_output));
         for j = 1:i
            r = src.mvops.dot(V(:,i1), V(:,j), 'precision_compute', precision_compute, 'precision_output', precision_output);
            V(:,i1) = precision_output(precision_compute(V(:,i1)) - precision_compute(r)*precision_compute(V(:,j)));
            if reorth
               r = src.mvops.dot(V(:,i1), V(:,j), 'precision_compute', precision_compute, 'precision_output', precision_output);
               V(:,i1) = precision_output(precision_compute(V(:,i1)) - precision_compute(r)*precision_compute(V(:,j)));
            end
         end

         t = src.mvops.norm2(V(:,i1), 'precision_compute', precision_compute, 'precision_output', precision_output);
         if t < select_tol
            tits = maxiter;
            break;
         else
            V(:,i1) = precision_output(precision_compute(V(:,i1)) / precision_compute(t));
         end
         tits = tits + 1;
      end
      % Compute Rayleigh-Ritz projection and restart
      [V, D] = src.rr.eig_rr(A, V, 'precision_matvec_compute', precision_compute, ...
                              'precision_matvec_output', precision_output, ...
                              'precision_eig', precision_eigenvalues, ...
                              'use_generalized', generalized, ...
                              'generalized_tol', generalized_tol, ...
                              'matvec', matvec);
      V = real(V);
      D = real(D);
      [~, idx] = sort(diag(D), 'descend');
      if tits < maxiter
         %% restart with the current eigenvector
         V(:, 1) = V(:, idx(1));
      else
         % return the first k eigenvectors
         V = V(:, idx(1:k));
         D = D(idx(1:k), idx(1:k));
      end
   end
end

function [precision_compute, precision_output, precision_eigenvalues, precision_A_matvec_compute, precision_A_matvec_output, seed, maxiter, kdim, reorth, generalized, generalized_tol, select_tol, matvec, check_params] = parse_options(varargin)
   if nargin > 1
      for i = 1:2:nargin
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'precision_eigenvalues'
               precision_eigenvalues = src.utils.parse_precision(varargin{i+1});
            case 'precision_A_matvec_compute'
               precision_A_matvec_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_A_matvec_output'
               precision_A_matvec_output = src.utils.parse_precision(varargin{i+1});
            case 'seed'
               seed = varargin{i+1};
            case 'maxiter'
               maxiter = varargin{i+1};;
            case 'kdim'
               kdim = varargin{i+1};
            case 'reorth'
               reorth = varargin{i+1};
            case 'generalized'
               generalized = varargin{i+1};
            case 'select_tol'
               select_tol = varargin{i+1};
            case 'generalized_tol'
               generalized_tol = varargin{i+1};
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
   if ~exist('precision_eigenvalues', 'var')
      precision_eigenvalues = @double;
   end
   if ~exist('precision_A_matvec_compute', 'var')
      precision_A_matvec_compute = precision_compute;
   end
   if ~exist('precision_A_matvec_output', 'var')
      precision_A_matvec_output = precision_A_matvec_compute;
   end
   if ~exist('seed', 'var')
      seed = 42;
   end
   if ~exist('maxiter', 'var')
      maxiter = 5;
   end
   if ~exist('kdim', 'var')
      kdim = maxiter;
   end
   if ~exist('reorth', 'var')
      reorth = true;
   end
   if ~exist('generalized', 'var')
      generalized = false;
   end
   if ~exist('select_tol', 'var')
      select_tol = src.utils.eps(char(precision_output));
   end
   if ~exist('generalized_tol', 'var')
      generalized_tol = 0.0;
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
      fprintf(' -- precision_eigenvalues: %s\n', char(precision_eigenvalues));
      fprintf(' -- precision_A_matvec_compute: %s\n', char(precision_A_matvec_compute));
      fprintf(' -- precision_A_matvec_output: %s\n', char(precision_A_matvec_output));
      fprintf(' -- seed: %d\n', seed);
      fprintf(' -- maxiter: %d\n', maxiter);
      fprintf(' -- kdim: %d\n', kdim);
      fprintf(' -- reorth: %d\n', reorth);
      fprintf(' -- generalized: %d\n', generalized);
      fprintf(' -- generalized_tol: %e\n', generalized_tol);
      fprintf(' -- select_tol: %e\n', select_tol);
      fprintf(' -- matvec: %s\n', char(matvec));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end