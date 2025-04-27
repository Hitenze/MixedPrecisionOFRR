function configs = test_configs()
% SVD_TEST_CONFIGS Returns configuration parameters for singular value computation tests
%
% Returns a structure containing multiple test parameter sets:
% - configs.ntests: Number of test configurations
% - configs.params{i}: Parameters for the i-th test configuration
%
% Each parameter set contains:
% - n1: Number of rows in the matrix
% - n2: Number of columns in the matrix
% - l: Gaussian kernel parameter
% - f: Gaussian kernel parameter
% - s: Gaussian kernel parameter
% - m: Subspace dimension
% - k: Number of singular values to compute
% - iter: Step size per iteration
% - maxiter: Maximum number of iterations
% - description: Brief description of the test configuration

% test settings

   configs = struct();
   
   % Set number of test configurations
   configs.ntests = 3;
   
   % Initialize parameter cell array
   configs.params = cell(1, configs.ntests);
   
   % Test configuration 1
   params1 = struct();
   params1.n1 = 1000;
   params1.n2 = 200;
   params1.l = 10;
   params1.f = 0.2;
   params1.s = 0e-01;
   params1.m = 20;
   params1.k = 10;
   params1.iter = 1;
   params1.maxiter = 10;
   params1.description = 'Rectangular matrix with strong decay (l=10)';
   configs.params{1} = params1;
   
   % Test configuration 2
   params2 = struct();
   params2.n1 = 1000;
   params2.n2 = 200;
   params2.l = 100;
   params2.f = 0.2;
   params2.s = 0e-01;
   params2.m = 10;
   params2.k = 4;
   params2.iter = 1;
   params2.maxiter = 5;
   params2.description = 'Rectangular matrix with weaker decay (l=20)';
   configs.params{2} = params2;
end 