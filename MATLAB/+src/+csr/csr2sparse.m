function [A] = csr2sparse(Acsr)
%% function A = csr2sparse(Acsr)
% Convert CSR matrix back to MATLAB sparse matrix.
%
% Inputs:
%   Acsr: CSR matrix struct
%      Acsr.nrows: number of rows
%      Acsr.ncols: number of columns
%      Acsr.i: length rows+1, start/end of each row
%      Acsr.j: length nnz, column index of each entry
%      Acsr.a: length nnz, value of each entry
% Outputs:
%   A: MATLAB sparse matrix
%
% Example:
%   A = src.csr.csr2sparse(Acsr);
%      Convert CSR matrix back to MATLAB sparse matrix

   nrows = Acsr.nrows;
   ncols = Acsr.ncols;
   nz = Acsr.i(nrows+1)-1;

   ii = zeros(nz,1);
   jj = zeros(nz,1);
   aa = zeros(nz,1);

   idx = 1;
   for i = 1:nrows
      i1 = Acsr.i(i);
      i2 = Acsr.i(i+1)-1;
      for j = i1:i2
         col = Acsr.j(j);
         val = Acsr.a_double(j);
         ii(idx) = i;
         jj(idx) = col;
         aa(idx) = val;
         idx = idx + 1;
      end
   end

   A = sparse(ii(1:idx-1),jj(1:idx-1),aa(1:idx-1),nrows,ncols);

end