function res=likeLink(f, R, str, varargin)
 
   
  if strcmp(str,'like')==1
      res=likelihood(f,R);
  elseif strcmp(str,'grad')==1
      res=gradient(f,R);
     
  elseif strcmp(str,'hess')==1
      res=hessian(f,R);     
  end
end

function like=likelihood(f,R)
    
     RelLike=log(1+exp(-R.* (f*f')));
     RelLike(R==0)=0;
     like = - sum(RelLike(:))/2;   
end

function grad=gradient(f,R)

    N=length(f);
    grad=zeros(N,1);

    for nn=1:N
        aa=R(nn,:)' .* f;
        aa(nn)=0;
        grad(nn) = sum(aa ./ (1+exp(f(nn)*aa)));
    end
    
end

function W=hessian(f,R)
    N=length(f);
    W=zeros(N);
    
    for ii=1:N
        for jj=ii:N
            
            rij=R(ii,jj);
            fij=f(ii)*f(jj);
            
            if rij~=0
                W(ii,jj)=(rij*(1+exp(rij*fij))-fij *exp(rij*fij))/(1+exp(rij*fij))^2;
            end
            W(jj,ii)=W(ii,jj);
        end
    end
    W=W+eye(N);
end