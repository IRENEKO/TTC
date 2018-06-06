function b=contract(a)
% b=contract(a)
% -------------
% Sums over all auxiliary indices of a Tensor Network a to return the
% underlying tensor b.
%
% b         =   tensor-matrix-vector-scalar, result after contraction,
%
% a         =   Tensor Network.
%
% Reference
% ---------
%
% A Tensor Network Kalman filter with an application in recursive MIMO Volterra system identification
% Fast and Accurate Tensor Completion with Tensor Trains: A System Identification Approach

% 2016, Kim Batselier, Zhongming Chen, Ngai Wong
% 2018, Ching-Yun KO

[d,n]=size(a.n);
b=reshape(a.core{1},[prod(a.n(1,1:n-1)) a.n(1,end)]);
sb=zeros(1,d*(n-2));
sb(1:n-2)=a.n(1,2:end-1);


for i=2:d
    b=b*reshape(a.core{i},[a.n(i,1) prod(a.n(i,2:end))]);
    sb((i-1)*(n-2)+1:i*(n-2))=a.n(i,2:end-1);
    if a.n(1)==1
        b=reshape(b,[prod(sb(1:(n-2)*i)) prod(size(b))/prod(sb(1:(n-2)*i))]);
    else
        b=reshape(b,[prod([a.n(1),sb(1:(n-2)*i)]) prod(size(b))/prod([a.n(1),sb(1:(n-2)*i)])]);
    end
end


if a.n(1)==1
    b=reshape(b,sb);
    I=zeros(1,d*(n-2));
    for i=1:n-2
        I((i-1)*d+1:i*d)=i:n-2:d*(n-2);
    end
else
    b=reshape(b,[a.n(1),sb]);
    I=zeros(1,d*(n-2));
    for i=1:n-2
        I((i-1)*d+1:i*d)=i:n-2:d*(n-2);
    end
    I=I+1;
    I=[1,I];
end


b=permute(b,I);

end
