function cores=mpsvd_op(core,n,r)
tensor=core.core;
[d,k]=size(n);
if ~isempty(core.r)
    tensor=reshape(tensor,[core.r(1)*n(1), n(2:end)']);
    indices=reshape([1:d*k],[d,k])';
    tensor=permute(tensor,indices(:));
    n2=prod(n,2)';
    tensor=reshape(tensor,[core.r(1)*n2(1),n2(2:end)]);
    cores=cell(1,d);
    r=[core.r(1),r,1];
else
    tensor=reshape(tensor,n(:)');
    indices=reshape([1:d*k],[d,k])';
    tensor=permute(tensor,indices(:));
    n2=prod(n,2)';
    tensor=reshape(tensor,n2);
    cores=cell(1,d);
    r=[1,r,1];
end

for i=d:-1:2
%     [U,S,V]=svd(reshape(tensor,[numel(tensor)/(n2(i)*r(i+1)),n2(i)*r(i+1)])','econ');
    [U,S,V]=svds(reshape(tensor,[numel(tensor)/(n2(i)*r(i+1)),n2(i)*r(i+1)])',r(i));
    cores{i}.r=[r(i) r(i+1)];
    cores{i}.n=n(i,:);
    cores{i}.core=reshape(U(:,1:r(i))',[r(i) n(i,:) r(i+1)]);
    tensor=reshape((S(1:r(i),1:r(i))*V(:,1:r(i))')',[r(1),n2(1:i-1),r(i)]);
end


cores{1}.r=[r(1) r(2)];
cores{1}.n=n(1,:);
cores{1}.core=reshape(tensor,[r(1) n(1,:) r(2)]);
end      