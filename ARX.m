function [M,Yin,beta] = ARX(q,k,x,u)

% q is the order of the exogenous part
% k is the lead time
% x is the output dataset (training)
% u is the input dataset (training)

% M is the [U] matrix, Y_hat = M*beta
% Yin are the initial values of Y matrix
% beta is the parameter matrix (which has been tuned)

dim_u = size(u);

% U Initialization
lq = q;
j = 0;
U = zeros(dim_u(1)-k-q+1,dim_u(2));
U = repmat(U,1,q);

%  U Computation
while (lq > 0)
    U(:,1+dim_u(2)*j:dim_u(2)*(j+1)) = [u(lq:end-k-j,:)];
    lq = lq-1;
    j = j+1;
end
clear lq

% beta tuning
M = [U];
Y = [x(q+k:end)];
beta = M\Y;
Yin = [x(1:q+k-1)];
