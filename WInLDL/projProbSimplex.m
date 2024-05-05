function X=projProbSimplex(y)

% u=sort(y,'descend');
% u=1-cumsum(u);
% j=1:length(u);
% 
% newu=u./j+y;
% rho=find(newu>0);
% rho=rho(end);
% 
% lambda=u(rho)/rho;
% 
% x=y+lambda;
% checkNonPosIndex=find(x<0);
% x(checkNonPosIndex)=0;


%%
% [N,D] = size(Y);
% X = sort(Y,2,'descend');
% Xtmp = (cumsum(X,2)-1) * diag(sparse(1./(1:D)));
% X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);

y = y';
X = max(bsxfun(@minus,y,max(bsxfun(@rdivide,cumsum(sort(y,1,'descend'),1)-1,(1:size(y,1))'),[],1)),0);
X = X';

end

