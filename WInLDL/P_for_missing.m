function Q = P_for_missing(P)
a = sum(P,2);    
A = P==0;        
c = ones(size(a)) - a;
b(b==0)=1;
d = c ./ b;
Q = A .* d + P;

