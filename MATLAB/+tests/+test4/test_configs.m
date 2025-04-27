function configs = test_configs()
% TEST_CONFIGS Returns configuration parameters for sparse matrix eigenvalue computation tests
%
% Returns a structure containing multiple test parameter sets:
% - configs.ntests: Number of test configurations
% - configs.params{i}: Parameters for the i-th test configuration
%
% Each parameter set contains:
% - matrix_name: Name of the sparse matrix from SuiteSparse collection
% - matrix_path: Path to the matrix data file
% - k: Number of eigenvalues to compute
% - kdim: Krylov subspace dimension
% - maxiter: Maximum number of iterations
% - description: Brief description of the test configuration

   configs = struct();
   
   % Set number of test configurations
   configs.ntests = 3;
   
   % Initialize parameter cell array
   configs.params = cell(1, configs.ntests);
   
   % Test configuration 1
   params1 = struct();
   params1.matrix_name = 'bcsstk01';
   params1.matrix_path = 'data/bcsstk01.mat';
   params1.k = 5;
   params1.kdim = 20;
   params1.maxiter = 20;
   params1.description = 'bcsstk01 matrix (48×48) from structural engineering';
   configs.params{1} = params1;
   
   % Test configuration 2
   params2 = struct();
   params2.matrix_name = 'bcsstk03';
   params2.matrix_path = 'data/bcsstk03.mat';
   params2.k = 10;
   params2.kdim = 50;
   params2.maxiter = 200;
   params2.description = 'bcsstk03 matrix (112×112) from structural engineering';
   configs.params{2} = params2;
   
   % Test configuration 3
   params3 = struct();
   params3.matrix_name = '1138_bus';
   params3.matrix_path = 'data/1138_bus.mat';
   params3.k = 20;
   params3.kdim = 100;
   params3.maxiter = 400;
   params3.description = '1138_bus matrix (1,138×1,138) from power systems';
   configs.params{3} = params3;
end 