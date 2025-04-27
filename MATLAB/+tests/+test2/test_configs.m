function configs = test_configs()
   % TEST_CONFIGS Returns configuration parameters for eigenvalue computation tests
   %
   % Returns a structure containing multiple test parameter sets:
   % - configs.ntests: Number of test configurations
   % - configs.params{i}: Parameters for the i-th test configuration
   %
   % Each parameter set contains:
   % - n: Matrix size
   % - l: Gaussian kernel parameter
   % - f: Gaussian kernel parameter
   % - s: Gaussian kernel parameter
   % - m: Subspace dimension
   % - k: Number of eigenvalues to compute
   % - iter: Step size per iteration
   % - maxiter: Maximum number of iterations
   % - description: Brief description of the test configuration

   configs = struct();
   
   % Set number of test configurations
   configs.ntests = 2;
   
   % Initialize parameter cell array
   configs.params = cell(1, configs.ntests);
   
   % Test configuration 1
   params1 = struct();
   params1.n = 1000;
   params1.l = 10;
   params1.f = 0.2;
   params1.s = 1e-01;
   params1.m = 50;
   params1.k = 20;
   params1.iter = 3;
   params1.maxiter = 10;
   params1.description = 'High subspace dimension, weaker decay';
   configs.params{1} = params1;
   
   % Test configuration 2
   params2 = struct();
   params2.n = 1000;
   params2.l = 100;
   params2.f = 0.2;
   params2.s = 0e-01;
   params2.m = 20;
   params2.k = 6;
   params2.iter = 2;
   params2.maxiter = 5;
   params2.description = 'Medium subspace dimension, strong decay';
   configs.params{2} = params2;
end