function distance = chebyshev(rd,pd)
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Chebyshev
%	
temp=abs(pd-rd);
temp=max(temp')';
distance=mean(temp);
end

