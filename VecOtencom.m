function [TN,e]=VecOtencom(y,u,r,init,varargin)
% [TN,e]=tencom(y,u,r,init,varargin)
% -------------
% Tensor completion given the inputs, outputs, TT-ranks, and TT
% initialization with vector outputs
%
% y         =   Outputs,
%
% u         =   Inputs,
%
% r         =   TT-ranks,
%
% init         =   TT initialization.
%
% Reference
% ---------
%
% Fast and Accurate Tensor Completion with Total Variation Regularized Tensor Trains

% 2018, Ching-Yun KO

N=size(u{1},1);
l=size(y,2);
% y=reshape(y',[N*l,1]);
y=y';
d=size(u,2);                 
for i=1:d
    n(i)=size(u{i},2);
end
r=[l r(:)' 1];                
MAXITR=3;
if ~isempty(varargin)
    fa=varargin{1};
else
    fa=1;
end


for i=1:d
    TN.core{i}=init{i}.core;
    TN.n(i,:)=[init{i}.r(1),init{i}.n,init{i}.r(2)];
end

Vp=cell(1,d);
Vm=cell(1,d);

if l==1
    Vm{1}=ones(N,1);
else
    Vm{1}=eye(l);
end
Vp{d}=ones(N,1);

for i=d:-1:2
  
    Vp{i-1}=dotkron(Vp{i},u{i})*reshape(permute(TN.core{i},[3 2 1]),[r(i+1)*n(i),r(i)]); 
end

% yhat=MOsim_tencom(u,TN);
% yhat=reshape(yhat',[N*l,1]);
% e(1)=norm(y(:)-yhat(:))/norm(y(:));

itr=1;                          % counts number of iterations
ltr=1;                          % flag that checks whether we sweep left to right
sweepindex=1;                   % index that indicates which TT core will be updated

% while itr<2 || ((e(itr) < e(itr-1)) && (itr < MAXITR) && e(itr) > THRESHOLD)
while itr<2 ||  (itr < MAXITR )
    updateTT;
    updatesweep;
    % only check residual after 1 half sweep
    if (sweepindex==d) || (sweepindex==1) % half a sweep
        itr=itr+1;
%         yhat=MOsim_tencom(u,TN);
%         yhat=reshape(yhat',[N*l,1]);
%         e(itr)=norm(y(:)-yhat(:))/norm(y(:));
    end    
end  

    function updateTT
%         % first construct the linear subsystem matrix
        if l==1
            A=dotkron(Vm{sweepindex},u{sweepindex},Vp{sweepindex});
        elseif sweepindex == 1
            A=dotkron(u{sweepindex},Vp{sweepindex});
        else            
            ind=randperm(N);
            ind=ind(1:floor(N/l*fa));
            Nn=length(ind);
            A=dotkron(Vm{sweepindex}(ind,:),u{sweepindex}(ind,:),Vp{sweepindex}(ind,:));
            A=reshape(A,[Nn,l,r(sweepindex)*n(sweepindex)*r(sweepindex+1)]);
            A=permute(A,[2 1 3]);
            A=reshape(A,[l*Nn,r(sweepindex)*n(sweepindex)*r(sweepindex+1)]);
        end 

        if ltr
            % left-to-right sweep, generate left orthogonal cores and update vk1
            if l==1
                g=pinv(A'*A)*(A'*y(:)); 
                [Q,R]=qr(reshape(g,[r(sweepindex)*(n(sweepindex)),r(sweepindex+1)])); 
%                 [Q,R]=qr(reshape(g,[r(sweepindex)*(n(sweepindex)),r(sweepindex+1)])); 
                TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
                TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n(sweepindex+1))*r(sweepindex+2)]),[r(sweepindex+1),n(sweepindex+1),r(sweepindex+2)]);
                Vm{sweepindex+1}=dotkron(Vm{sweepindex},u{sweepindex})*reshape(TN.core{sweepindex},[r(sweepindex)*n(sweepindex),r(sweepindex+1)]); % N x r_{i}                         
            elseif sweepindex==1
                g=pinv(A'*A)*(A'*y');
                [Q,R]=qr(reshape(g',[r(sweepindex)*(n(sweepindex)),r(sweepindex+1)])); 
                TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
                TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n(sweepindex+1))*r(sweepindex+2)]),[r(sweepindex+1),n(sweepindex+1),r(sweepindex+2)]);
                Vm{sweepindex+1}=u{sweepindex}*reshape(permute(TN.core{sweepindex},[2 1 3]),[n(sweepindex),r(sweepindex)*r(sweepindex+1)]); %N x r_{i-1}r_i
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,r(sweepindex)*r(sweepindex+1)]);                
            else
                yhat=y(:,ind);
                g=pinv(A'*A)*(A'*yhat(:));
                [Q,R]=qr(reshape(g,[r(sweepindex)*(n(sweepindex)),r(sweepindex+1)])); 
                TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
                TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n(sweepindex+1))*r(sweepindex+2)]),[r(sweepindex+1),n(sweepindex+1),r(sweepindex+2)]);
                Vm{sweepindex+1}=reshape(dotkron(Vm{sweepindex},u{sweepindex}),[N*l,r(sweepindex)*n(sweepindex)])*reshape(TN.core{sweepindex},[r(sweepindex)*n(sweepindex),r(sweepindex+1)]);                
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,l*r(sweepindex+1)]);
            end
        else
            % right-to-left sweep, generate right orthogonal cores and update vk2
            yhat=y(:,ind);
            g=pinv(A'*A)*(A'*yhat(:)); 
            [Q,R]=qr(reshape(g,[r(sweepindex),(n(sweepindex))*r(sweepindex+1)])'); 
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex))',[r(sweepindex),n(sweepindex),r(sweepindex+1)]);
            TN.core{sweepindex-1}=reshape(reshape(TN.core{sweepindex-1},[r(sweepindex-1)*(n(sweepindex-1)),r(sweepindex)])*R(1:r(sweepindex),:)',[r(sweepindex-1),n(sweepindex-1),r(sweepindex)]);
            Vp{sweepindex-1}=dotkron(Vp{sweepindex},u{sweepindex})*reshape(permute(TN.core{sweepindex},[3 2 1]),[r(sweepindex+1)*n(sweepindex),r(sweepindex)]); % N x r_{i-1}    

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