function [Acsr] = sparse2csr(A)
%% function Acsr = sparse2csr(A)
% Convert MATLAB sparse matrix to CSR format.
%
% Inputs:
%   A: MATLAB sparse matrix
% Outputs:
%   Acsr: CSR matrix struct
%      Acsr.nrows: number of rows
%      Acsr.ncols: number of columns
%      Acsr.i: length rows+1, start/end of each row
%      Acsr.j: length nnz, column index of each entry
%      Acsr.a_double: length nnz, value of each entry in double precision
%      Acsr.a_single: length nnz, value of each entry in single precision
%      Acsr.a_half: length nnz, value of each entry in half precision
%
% Example:
%   Acsr = src.csr.sparse2csr(A);
%      Convert MATLAB sparse matrix to CSR format

   assert(all(class(A) == 'double'), 'A must be a double precision matrix');
   assert(issparse(A), 'A must be a sparse matrix');

   [nrows, ncols] = size(A);
   nz = nnz(A);

   Acsr.nrows = nrows;
   Acsr.ncols = ncols;

   Acsr.i = zeros(nrows+1,1);
   Acsr.j = zeros(nz,1);
   Acsr.a_double = zeros(nz,1);

   AT = A';

   idx = 1;
   for i = 1:nrows
      Acsr.i(i) = idx;
      [jj,~,aa] = find(AT(:,i));
      for j = 1:length(jj)
         Acsr.j(idx) = jj(j);
         Acsr.a_double(idx) = aa(j);
         idx = idx + 1;
      end
   end
   Acsr.i(nrows+1) = idx;

   Acsr.a_single = single(Acsr.a_double);
   Acsr.a_half = half(Acsr.a_double);
   
end

