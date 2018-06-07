function simpic=repro_exp1(data,para,version)
h=para.h;    % scaling factor h 
maxitr=para.maxitr;
if version==2
    lambda=para.lambda;
    idf=para.idf;
end
R2=para.R2;
R=para.R;
Rd=para.Rd;
n=para.n;   
mi=para.mi;
kn=para.kn;
Mi=para.mi;
Kn=para.Kn;

szo=size(data);
Aref=reshape(data,[prod(szo),1]);
yref=double(Aref(Mi));
A=reshape(data,[prod(szo(1:2)),szo(3)]);
A(mi,:)=0;
A=reshape(A,szo);
d=length(n);

% For images
r=[R2,R*ones(1,d-4),Rd,3];


%% Appending zeros and reconstruct the identifier mi kn Mi Kn (skip if don't need)
A=[A;zeros(3,szo(2),3)];
A=[A,zeros(szo(1)+3,5,3)];  
sz=size(A);
tempmi=ceil(mi/size(data,1));
tempkn=ceil(kn/size(data,1));
for hh=2:size(data,2)
    id=find(tempmi==hh);
    mi(id)=mi(id)+(hh-1)*3;
    id=find(tempkn==hh);
    kn(id)=kn(id)+(hh-1)*3;
end
Mi=mi;
Kn=kn;
for i=1:ndims(data)-1
    Mi=[Mi,i*numel(A)/3+mi];
    Kn=[Kn,i*numel(A)/3+kn];
end
clear tempmi tempkn id 


%% The initialization of the tensor train and the construction of the inputs 
tic;
init=imresize(A,[floor(sz(1)/h),floor(sz(2)/h)],'box');
init=imresize(init,[sz(1),sz(2)]);
Init.core=double(init);
Init.r=[1,1];
clear init
if version==1
    init=mpsvd(Init,n',r);
elseif version==2
    init=mpsvd_op(Init,n',r);
end
clear Init
A=A(:);
A=double(A);

temp=[1:prod(sz)]';  
for k=d:-1:2
    index(:,k)=ceil(temp/(prod(n(1:k-1)))); 
    temp=temp-(index(:,k)-1)*(prod(n(1:k-1)));
end
index(:,1)=temp;
clear temp

for k=1:d
    u{k}=zeros(size(Kn,2),n(k));
    for i=1:size(Kn,2)
        u{k}(i,index(Kn(i),k))=1;
    end
end
for i=1:size(Kn,2)
    y(i)=A(Kn(i));
end
Initial_time=toc


%% Teneosr completion
tic;
if version==1
    TN=tencom(y',u,r,init,maxitr);
elseif version==2
    TN=tencom_TV(y',u,r,init,Kn,idf,lambda);
end
temp=contract(TN);
temp=temp(:);
ysim=temp(Mi);
clear temp
Completion_time=toc

A(Mi)=ysim;
simpic=reshape(A,sz);
simpic=simpic(1:szo(1),1:szo(2),:);
Aref=reshape(Aref,szo);                                                               
simpic=uint8(simpic);
RSE=sqrt(sum((double(simpic(:))-double(Aref(:))).^2))/sqrt(sum(double(Aref(:)).^2))

end