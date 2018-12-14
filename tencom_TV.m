function [TN,e]=tencom_TV(y,u,r,init,Kn,idf,lamda,varargin)
% [TN,e]=tencom(y,u,r,init,varargin)
% -------------
% Tensor completion given the inputs, outputs, TT-ranks, and TT
% initialization, coordinate of known entries, dimensions identifier, and 
% TV term parameter lambda with scalar inputs
%
% y         =   Outputs,
%
% u         =   Inputs,
%
% r         =   TT-ranks,
%
% init      =   TT initialization,
%
% Kn        =   Known entris' indices.
%
% idf       =   dimensions identifier,
%
% lambda    =   TV term parameter.
%
% Reference
% ---------
%
% Fast and Accurate Tensor Completion with Total Variation Regularized Tensor Trains


% 2018, Ching-Yun KO
N=size(u{1},1);
d=size(u,2);                 
for i=1:d
    n(i)=size(u{i},2);
end
r=[1 r(:)' 1];                
if ~isempty(varargin)
    MAXITR=varargin{1}+1;
else
    MAXITR=2+1;
end

fa=2;

for i=1:d
    TN.core{i}=init{i}.core;
    TN.n(i,:)=[init{i}.r(1),init{i}.n,init{i}.r(2)];
end

for i=1:length(idf)-1
    Ddim{i}=prod(n(idf(i):idf(i+1)-1));
    temp.core=[toeplitz([1,zeros(1,Ddim{i}-2)],[1,-1,zeros(1,Ddim{i}-2)]);zeros(1,Ddim{i})]';
    temp.r=[];
    DD=mpsvd(temp,n(idf(i):idf(i+1)-1)'*[1,1],3*ones(1,d-1));
    for j=1:idf(i)-1
        D{i}{j}=reshape(eye(n(j)),[1,n(j),n(j),1]);
    end
    for j=1:idf(i+1)-idf(i)
        D{i}{idf(i)+j-1}=DD{j}.core;
    end
    for j=idf(i+1):d
        D{i}{j}=reshape(eye(n(j)),[1,n(j),n(j),1]);
    end 
end


Vp=cell(1,d);
Vm=cell(1,d);
Vm{1}=ones(N,1);
Vp{d}=ones(N,1);

Dm{1}{1}=1;
Dm{2}{1}=1;

for j=d:-1:2
    Vp{j-1}=dotkron(Vp{j},u{j})*reshape(permute(TN.core{j},[3 2 1]),[r(j+1)*n(j),r(j)]); 
end

% for first lamda term 
% Dm and Dp are dofferent from Vm and Vp because they're already ans'*ans
for j=d:-1:idf(2)
    Dp{1}{j-1}=eye(r(j));
end
sz=size(D{1}{idf(2)-1});
Dp{1}{idf(2)-2}=reshape(permute(D{1}{idf(2)-1},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{idf(2)-1},[2,1,3]),[n(idf(2)-1),r(idf(2)-1)*r(idf(2))]);
temp=reshape(permute(reshape(Dp{1}{idf(2)-2},[sz(1),sz(3),r(idf(2)-1),r(idf(2))]),[2,4,1,3]),[sz(3)*r(idf(2)),sz(1)*r(idf(2)-1)]);
Dp{1}{idf(2)-2}=reshape(temp'*temp,[sz(1)*r(idf(2)-1),sz(1)*r(idf(2)-1)]);
for j=idf(2)-2:-1:2
    sz=size(D{1}{j});
    Dp{1}{j-1}=reshape(permute(D{1}{j},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{j},[2,1,3]),[n(j),r(j)*r(j+1)]);
    temp=reshape(permute(reshape(Dp{1}{j-1},[sz(1),sz(3),sz(4),r(j),r(j+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(j)*sz(4)*r(j+1)]);
    Dp{1}{j-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(j),sz(4)*r(j+1),sz(1)*r(j),sz(4)*r(j+1)]),[1,3,2,4]),[sz(1)*r(j)*sz(1)*r(j),sz(4)*r(j+1)*sz(4)*r(j+1)])*Dp{1}{j}(:);
end


% for second lamda term 
for j=d
    Dp{2}{j-1}=eye(r(j));
end
sz=size(D{2}{idf(3)-1});
Dp{2}{idf(3)-2}=reshape(permute(D{2}{idf(3)-1},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{idf(3)-1},[2,1,3]),[n(idf(3)-1),r(idf(3)-1)*r(idf(3))]);
temp=reshape(permute(reshape(Dp{2}{idf(3)-2},[sz(1),sz(3),r(idf(3)-1),r(idf(3))]),[2,4,1,3]),[sz(3)*r(idf(3)),sz(1)*r(idf(3)-1)]);
Dp{2}{idf(3)-2}=reshape(temp'*temp,[sz(1)*r(idf(3)-1),sz(1)*r(idf(3)-1)]);
for j=idf(3)-2:-1:idf(2)
    sz=size(D{2}{j});
    Dp{2}{j-1}=reshape(permute(D{2}{j},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{j},[2,1,3]),[n(j),r(j)*r(j+1)]);
    temp=reshape(permute(reshape(Dp{2}{j-1},[sz(1),sz(3),sz(4),r(j),r(j+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(j)*sz(4)*r(j+1)]);
    Dp{2}{j-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(j),sz(4)*r(j+1),sz(1)*r(j),sz(4)*r(j+1)]),[1,3,2,4]),[sz(1)*r(j)*sz(1)*r(j),sz(4)*r(j+1)*sz(4)*r(j+1)])*Dp{2}{j}(:);
end
for j=idf(2)-1:-1:2
    sz=size(D{2}{j});
    Dp{2}{j-1}=reshape(permute(D{2}{j},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{j},[2,1,3]),[n(j),r(j)*r(j+1)]);
    temp=reshape(permute(reshape(Dp{2}{j-1},[sz(1),sz(3),r(j),r(j+1)]),[2,1,3,4]),[sz(3),sz(1)*r(j)*r(j+1)]);
    Dp{2}{j-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(j),r(j+1),sz(1),r(j)*r(j+1)]),[1,3,2,4]),[(sz(1))^2*(r(j))^2,(r(j+1))^2])*Dp{2}{j}(:);
end


itr=1;                          % counts number of iterations
ltr=1;                          % flag that checks whether we sweep left to right
sweepindex=1;                   % index that indicates which TT core will be updated

% temp=contract(TN);
% temp=temp(:);
% e(itr)=norm(y-temp(Kn))/norm(y);
while itr<2 ||  (itr < MAXITR )
    updateTT;
    updatesweep;
    % only check residual after 1 half sweep
    if (sweepindex==d) || (sweepindex==1) % half a sweep
        itr=itr+1;
        if itr==2
            temp=contract(TN);
            temp=temp(:);
            e(itr)=norm(y-temp(Kn))/norm(y);
            lamda=lamda*e(itr);
        end
    end    
end  

    function updateTT
%         % first construct the linear subsystem matrix
%         A=dotkron(Vm{sweepindex},u{sweepindex},Vp{sweepindex});
        
%         if l==1
%             A=dotkron(Vm{sweepindex},u{sweepindex},Vp{sweepindex});
%         elseif sweepindex == 1
%             A=dotkron(u{sweepindex},Vp{sweepindex});
%         else            
            ind=randperm(N);
            ind=ind(1:floor(N/fa));
            Nn=length(ind);
            A=dotkron(Vm{sweepindex}(ind,:),u{sweepindex}(ind,:),Vp{sweepindex}(ind,:));
            A=reshape(A,[Nn,r(sweepindex)*n(sweepindex)*r(sweepindex+1)]);
%         end 
        
        
        sz1=size(D{1}{sweepindex});
        if length(sz1)==3
            sz1=[sz1,1];            
        end
        sz2=size(D{2}{sweepindex});
        if length(sz2)==3
            sz2=[sz2,1];            
        end
        if sweepindex==1
            % first lamda
            down=reshape(permute(reshape(Dp{1}{sweepindex},[sz1(4),r(sweepindex+1),sz1(4),r(sweepindex+1)]),[1,3,2,4]),[(sz1(4))^2,(r(sweepindex+1))^2]);
            temp=reshape(permute(reshape(D{1}{sweepindex},sz1),[1,2,4,3]),[sz1(1)*sz1(2)*sz1(4),sz1(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz1(1),sz1(2),sz1(4),sz1(1),sz1(2),sz1(4)]),[1,4,2,5,3,6]),[(sz1(1))^2*(sz1(2))^2,(sz1(4))^2]);
            W1=reshape(temp*down,[r(sweepindex),r(sweepindex),sz1(2),sz1(2),r(sweepindex+1),r(sweepindex+1)]);
            W1=reshape(permute(W1,[1,3,5,2,4,6]),[r(sweepindex)*sz1(2)*r(sweepindex+1),r(sweepindex)*sz1(2)*r(sweepindex+1)]);
            % second lamda
            down=reshape(permute(reshape(Dp{2}{sweepindex},[sz2(4),r(sweepindex+1),sz2(4),r(sweepindex+1)]),[1,3,2,4]),[(sz2(4))^2,(r(sweepindex+1))^2]);
            temp=reshape(permute(reshape(D{2}{sweepindex},sz2),[1,2,4,3]),[sz2(1)*sz2(2)*sz2(4),sz2(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz2(1),sz2(2),sz2(4),sz2(1),sz2(2),sz2(4)]),[1,4,2,5,3,6]),[(sz2(1))^2*(sz2(2))^2,(sz2(4))^2]);
            W2=reshape(temp*down,[r(sweepindex),r(sweepindex),sz2(2),sz2(2),r(sweepindex+1),r(sweepindex+1)]);
            W2=reshape(permute(W2,[1,3,5,2,4,6]),[r(sweepindex)*sz2(2)*r(sweepindex+1),r(sweepindex)*sz2(2)*r(sweepindex+1)]);
        elseif sweepindex==d
            % first lamda
            temp=reshape(permute(reshape(D{1}{sweepindex},sz1),[1,2,4,3]),[sz1(1)*sz1(2)*sz1(4),sz1(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz1(1),sz1(2),sz1(4),sz1(1),sz1(2),sz1(4)]),[1,4,2,5,3,6]),[(sz1(1))^2,(sz1(2))^2*(sz1(4))^2]);
            up=reshape(permute(reshape(Dm{1}{sweepindex},[sz1(1),r(sweepindex),sz1(1),r(sweepindex)]),[2,4,1,3]),[(r(sweepindex))^2,(sz1(1))^2]);
            W1=reshape(up*temp,[r(sweepindex),r(sweepindex),sz1(2),sz1(2),r(sweepindex+1),r(sweepindex+1)]);
            W1=reshape(permute(W1,[1,3,5,2,4,6]),[r(sweepindex)*sz1(2)*r(sweepindex+1),r(sweepindex)*sz1(2)*r(sweepindex+1)]);
            % second lamda
            temp=reshape(permute(reshape(D{2}{sweepindex},sz2),[1,2,4,3]),[sz2(1)*sz2(2)*sz2(4),sz2(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz2(1),sz2(2),sz2(4),sz2(1),sz2(2),sz2(4)]),[1,4,2,5,3,6]),[(sz2(1))^2,(sz2(2))^2*(sz2(4))^2]);
            up=reshape(permute(reshape(Dm{2}{sweepindex},[sz2(1),r(sweepindex),sz2(1),r(sweepindex)]),[2,4,1,3]),[(r(sweepindex))^2,(sz2(1))^2]);
            W2=reshape(up*temp,[r(sweepindex),r(sweepindex),sz2(2),sz2(2),r(sweepindex+1),r(sweepindex+1)]);
            W2=reshape(permute(W2,[1,3,5,2,4,6]),[r(sweepindex)*sz2(2)*r(sweepindex+1),r(sweepindex)*sz2(2)*r(sweepindex+1)]);
        else
            % first lamda
            down=reshape(permute(reshape(Dp{1}{sweepindex},[sz1(4),r(sweepindex+1),sz1(4),r(sweepindex+1)]),[1,3,2,4]),[(sz1(4))^2,(r(sweepindex+1))^2]);
            temp=reshape(permute(reshape(D{1}{sweepindex},sz1),[1,2,4,3]),[sz1(1)*sz1(2)*sz1(4),sz1(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz1(1),sz1(2),sz1(4),sz1(1),sz1(2),sz1(4)]),[1,4,2,5,3,6]),[(sz1(1))^2,(sz1(2))^2*(sz1(4))^2]);
            up=reshape(permute(reshape(Dm{1}{sweepindex},[sz1(1),r(sweepindex),sz1(1),r(sweepindex)]),[2,4,1,3]),[(r(sweepindex))^2,(sz1(1))^2]);
            W1=reshape(reshape(up*temp,[(r(sweepindex))^2*(sz1(2))^2,(sz1(4))^2])*down,[r(sweepindex),r(sweepindex),sz1(2),sz1(2),r(sweepindex+1),r(sweepindex+1)]);
            W1=reshape(permute(W1,[1,3,5,2,4,6]),[r(sweepindex)*sz1(2)*r(sweepindex+1),r(sweepindex)*sz1(2)*r(sweepindex+1)]);
            % second lamda
            down=reshape(permute(reshape(Dp{2}{sweepindex},[sz2(4),r(sweepindex+1),sz2(4),r(sweepindex+1)]),[1,3,2,4]),[(sz2(4))^2,(r(sweepindex+1))^2]);
            temp=reshape(permute(reshape(D{2}{sweepindex},sz2),[1,2,4,3]),[sz2(1)*sz2(2)*sz2(4),sz2(3)]);
            temp=reshape(permute(reshape(temp*temp',[sz2(1),sz2(2),sz2(4),sz2(1),sz2(2),sz2(4)]),[1,4,2,5,3,6]),[(sz2(1))^2,(sz2(2))^2*(sz2(4))^2]);
            up=reshape(permute(reshape(Dm{2}{sweepindex},[sz2(1),r(sweepindex),sz2(1),r(sweepindex)]),[2,4,1,3]),[(r(sweepindex))^2,(sz2(1))^2]);
            W2=reshape(reshape(up*temp,[(r(sweepindex))^2*(sz2(2))^2,(sz2(4))^2])*down,[r(sweepindex),r(sweepindex),sz2(2),sz2(2),r(sweepindex+1),r(sweepindex+1)]);
            W2=reshape(permute(W2,[1,3,5,2,4,6]),[r(sweepindex)*sz2(2)*r(sweepindex+1),r(sweepindex)*sz2(2)*r(sweepindex+1)]);
        end
        yhat=y(ind);
        g=pinv(A'*A+lamda(1)*W1'*W1+lamda(2)*W2'*W2)*(A'*yhat(:));
%         g=pinv(A'*A+lamda(1)*W1'*W1+lamda(2)*W2'*W2)*(A'*y);

        if ltr
            % left-to-right sweep, generate left orthogonal cores and update vk1
            [Q,R]=qr(reshape(g,[r(sweepindex)*(n(sweepindex)),r(sweepindex+1)])); 
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
            TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n(sweepindex+1))*r(sweepindex+2)]),[r(sweepindex+1),n(sweepindex+1),r(sweepindex+2)]);
            Vm{sweepindex+1}=dotkron(Vm{sweepindex},u{sweepindex})*reshape(TN.core{sweepindex},[r(sweepindex)*n(sweepindex),r(sweepindex+1)]); % N x r_{i}
            % first lamda
            sz=size(D{1}{sweepindex});
            if length(sz)==3
                sz=[sz,1];              
            end
            Dm{1}{sweepindex+1}=reshape(permute(D{1}{sweepindex},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{sweepindex},[2,1,3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]);
            temp=reshape(permute(reshape(Dm{1}{sweepindex+1},[sz(1),sz(3),sz(4),r(sweepindex),r(sweepindex+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(sweepindex)*sz(4)*r(sweepindex+1)]);
            Dm{1}{sweepindex+1}=Dm{1}{sweepindex}(:)'*reshape(permute(reshape(temp'*temp,[sz(1)*r(sweepindex),sz(4)*r(sweepindex+1),sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)]),[1,3,2,4]),[sz(1)*r(sweepindex)*sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)*sz(4)*r(sweepindex+1)]);
            % second lamda
            if sweepindex <= idf(2)-1
                Dm{2}{sweepindex+1}=eye(r(sweepindex+1));                
            else
                sz=size(D{2}{sweepindex});
                if length(sz)==3
                sz=[sz,1];              
                end
                Dm{2}{sweepindex+1}=reshape(permute(D{2}{sweepindex},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{sweepindex},[2,1,3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]);
                temp=reshape(permute(reshape(Dm{2}{sweepindex+1},[sz(1),sz(3),sz(4),r(sweepindex),r(sweepindex+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(sweepindex)*sz(4)*r(sweepindex+1)]);
                Dm{2}{sweepindex+1}=Dm{2}{sweepindex}(:)'*reshape(permute(reshape(temp'*temp,[sz(1)*r(sweepindex),sz(4)*r(sweepindex+1),sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)]),[1,3,2,4]),[sz(1)*r(sweepindex)*sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)*sz(4)*r(sweepindex+1)]);
            end
        else
            % right-to-left sweep, generate right orthogonal cores and update vk2
            [Q,R]=qr(reshape(g,[r(sweepindex),(n(sweepindex))*r(sweepindex+1)])'); 
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex))',[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
            TN.core{sweepindex-1}=reshape(reshape(TN.core{sweepindex-1},[r(sweepindex-1)*(n(sweepindex-1)),r(sweepindex)])*R(1:r(sweepindex),:)',[r(sweepindex-1),n(sweepindex-1),r(sweepindex)]);
            Vp{sweepindex-1}=dotkron(Vp{sweepindex},u{sweepindex})*reshape(permute(TN.core{sweepindex},[3 2 1]),[r(sweepindex+1)*n(sweepindex),r(sweepindex)]); % N x r_{i-1}   
            % first lamda
            if sweepindex >= idf(2)
                Dp{1}{sweepindex-1}=eye(r(sweepindex));      
            elseif sweepindex == idf(2)-1
                sz=size(D{1}{idf(2)-1});
                Dp{1}{idf(2)-2}=reshape(permute(D{1}{idf(2)-1},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{idf(2)-1},[2,1,3]),[n(idf(2)-1),r(idf(2)-1)*r(idf(2))]);
                temp=reshape(permute(reshape(Dp{1}{idf(2)-2},[sz(1),sz(3),r(idf(2)-1),r(idf(2))]),[2,4,1,3]),[sz(3)*r(idf(2)),sz(1)*r(idf(2)-1)]);
                Dp{1}{idf(2)-2}=reshape(temp'*temp,[sz(1)*r(idf(2)-1),sz(1)*r(idf(2)-1)]);
            else
                sz=size(D{1}{sweepindex});
                Dp{1}{sweepindex-1}=reshape(permute(D{1}{sweepindex},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{sweepindex},[2,1,3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]);
                temp=reshape(permute(reshape(Dp{1}{sweepindex-1},[sz(1),sz(3),sz(4),r(sweepindex),r(sweepindex+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(sweepindex)*sz(4)*r(sweepindex+1)]);
                Dp{1}{sweepindex-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(sweepindex),sz(4)*r(sweepindex+1),sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)]),[1,3,2,4]),[sz(1)*r(sweepindex)*sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)*sz(4)*r(sweepindex+1)])*Dp{1}{sweepindex}(:);
            end 
            % second lamda
            if sweepindex == d
                Dp{2}{sweepindex-1}=eye(r(sweepindex));
            elseif sweepindex == idf(3)-1
                sz=size(D{2}{sweepindex});
                Dp{2}{idf(3)-2}=reshape(permute(D{2}{idf(3)-1},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{idf(3)-1},[2,1,3]),[n(idf(3)-1),r(idf(3)-1)*r(idf(3))]);
                temp=reshape(permute(reshape(Dp{2}{idf(3)-2},[sz(1),sz(3),r(idf(3)-1),r(idf(3))]),[2,4,1,3]),[sz(3)*r(idf(3)),sz(1)*r(idf(3)-1)]);
                Dp{2}{idf(3)-2}=reshape(temp'*temp,[sz(1)*r(idf(3)-1),sz(1)*r(idf(3)-1)]);
            elseif sweepindex >= idf(2)
                sz=size(D{2}{sweepindex});
                Dp{2}{sweepindex-1}=reshape(permute(D{2}{sweepindex},[1,3,4,2]),[sz(1)*sz(3)*sz(4),sz(2)])*reshape(permute(TN.core{sweepindex},[2,1,3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]);
                temp=reshape(permute(reshape(Dp{2}{sweepindex-1},[sz(1),sz(3),sz(4),r(sweepindex),r(sweepindex+1)]),[2,1,4,3,5]),[sz(3),sz(1)*r(sweepindex)*sz(4)*r(sweepindex+1)]);
                Dp{2}{sweepindex-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(sweepindex),sz(4)*r(sweepindex+1),sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)]),[1,3,2,4]),[sz(1)*r(sweepindex)*sz(1)*r(sweepindex),sz(4)*r(sweepindex+1)*sz(4)*r(sweepindex+1)])*Dp{2}{sweepindex}(:);
            else
                sz=size(D{2}{sweepindex});
                Dp{2}{sweepindex-1}=reshape(permute(D{2}{sweepindex},[1,3,2]),[sz(1)*sz(3),sz(2)])*reshape(permute(TN.core{sweepindex},[2,1,3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]);
                temp=reshape(permute(reshape(Dp{2}{sweepindex-1},[sz(1),sz(3),r(sweepindex),r(sweepindex+1)]),[2,1,3,4]),[sz(3),sz(1)*r(sweepindex)*r(sweepindex+1)]);
                Dp{2}{sweepindex-1}=reshape(permute(reshape(temp'*temp,[sz(1)*r(sweepindex),r(sweepindex+1),sz(1),r(sweepindex)*r(sweepindex+1)]),[1,3,2,4]),[(sz(1))^2*(r(sweepindex))^2,(r(sweepindex+1))^2])*Dp{2}{sweepindex}(:);
            end
        end
    end


    function updatesweep
        if ltr
            sweepindex=sweepindex+1;
            if sweepindex== d                
                ltr=0;
            end
        else
            sweepindex=sweepindex-1;
            if sweepindex== 1                
                ltr=1;
            end
        end
    end
end
