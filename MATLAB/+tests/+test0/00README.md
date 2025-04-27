# Test 0: Precision Impact on Eigenvalue Computation

We uses three different precision for 
   - Matvec with A
   - QR factorization
   - Rayleigh-Ritz (always solve final problem in double precision, but the A*V in V'*A*V is computed in the precision of Matvec with A)
