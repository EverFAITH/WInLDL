function Q = D_for_missing(D)
a = sum(D,2);    
b = sum(D==0,2);   
c = ones(size(a)) - a;
b0 = b;
b(b==0)=1;
d = c ./ b;
A = d .* (D==0);
B = (b0==1) .* A;
Q = D + B;