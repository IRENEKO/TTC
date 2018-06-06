function cores=mpsvd(core,n,r)
tensor=core.core;
[d,k]=size(n);
if ~isempty(core.r)
    tensor=reshape(tensor,[core.r(1)*n(1), n(2:end)']);
    indices=reshape([1:d*k],[d,k])';
    tensor=permute(tensor,indices(:));
    n2=prod(n,2)';
    tensor=reshape(tensor,[core.r(1)*n2(1),n2(2:end)]);
    cores=cell(1,d);
    r=[core.r(1),r];
else
    tensor=reshape(tensor,n(:)');
    indices=reshape([1:d*k],[d,k])';
    tensor=permute(tensor,indices(:));
    n2=prod(n,2)';
    tensor=reshape(tensor,n2);
    cores=cell(1,d);
    r=[1,r];
end


for i=1:d-1
%     [U,S,V]=svd(reshape(tensor,[r(i)*n2(i),numel(tensor)/(r(i)*n2(i))]),'econ');
    [U,S,V]=svds(reshape(tensor,[r(i)*n2(i),numel(tensor)/(r(i)*n2(i))]),r(i+1));
    cores{i}.r=[r(i) r(i+1)];
    cores{i}.n=n(i,:);
    
    r(i+1)=min(rank(S),r(i+1));
    
    cores{i}.core=reshape(U(:,1:r(i+1)),[r(i) n(i,:) r(i+1)]);
    tensor=reshape(S(1:r(i+1),1:r(i+1))*V(:,1:r(i+1))',[r(i+1),n2(i+1:end)]);
end
cores{d}.r=[r(i+1) 1];
cores{d}.n=n(end,:);
cores{d}.core=reshape(tensor,[r(i+1) n(end,:) 1]);
end       