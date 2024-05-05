function distance = cosine(rd,pd)
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Cosine
%	
inner=sum(pd.*rd,2);
len=(sqrt(sum(pd.*pd,2))).*(sqrt(sum(rd.*rd,2)));
distance=mean(inner./len);
end


