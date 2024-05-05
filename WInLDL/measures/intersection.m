function similarity=intersection(rd,pd)
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       SIMILARITY: the average of intersection
%
temp=sum(min(pd,rd),2);
similarity=mean(temp);
end