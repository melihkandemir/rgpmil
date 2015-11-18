function res=likeBinMILClass(f, y, str, varargin)
 
  indices=varargin{1}{1};
  
  
  if strcmp(str,'like')==1
      res=likelihood(f,y,indices);
  elseif strcmp(str,'grad')==1
      res=gradient(f,y,indices);
    %  res2=take_gradient(@likelihood,f,y,indices);
      aa=0;
  elseif strcmp(str,'hess')==1
      res=hessian(f,y,indices);
   %   res2=take_hessian(@gradient,f,y,indices);
      aa=0;
  end
end

function like=likelihood(f,y,indices)
    B=max(indices);
    like=0;

    for bb=1:B
       fb=f(indices==bb);
       like = like - log(  1+exp(logsumexp(fb))^(-y(bb))  );
    end
end

function grad=gradient(f,y,indices)

    B=max(indices);
    grad=zeros(length(f),1);

    for bb=1:B
       Nb=sum(indices==bb);
       fb=f(indices==bb);
       gradbb=zeros(Nb,1);

       for nn=1:Nb
         gradbb(nn)=y(bb)*sum(exp(fb))^(-y(bb)-1)*exp(logsumexp(fb(nn)))/( 1+exp(logsumexp(fb))^(-y(bb)) );
       end

       grad(indices==bb)=gradbb;
    end

end

function W=hessian(f,y,indices)
    N=length(f);
    W=zeros(N);
    
    for ii=1:N
        for jj=ii:N
            
            if indices(jj) ~=indices(ii)
                continue;
            end
            
            fb=f(indices==indices(ii));
            sumE=exp(logsumexp(fb));
            expF=exp(f(ii));
            expFJ=exp(f(jj));
            yF=y(indices(ii));

            numer=expF*yF*sumE^(-yF-1);
            denom=1+sumE^(-yF);
            
            if ii==jj % the diagonal
                
                numerDer=yF*expF*sumE^(-yF-1)+expF^2*(-yF-1)*sumE^(-yF-2);
                denomDer=-yF*expF*sumE^(-yF-1);
                
                
            else
                numerDer=yF*expF*(-yF-1)*sumE^(-yF-2)*expFJ;
                denomDer=-yF*sumE^(-yF-1)*expFJ;                 
                
                %numerDer=0;
                %numer=0;
               
            end                        
            
            W(ii,jj)=(numerDer*denom-numer*denomDer)/(denom^2);
            
            W(jj,ii)=W(ii,jj);
        end
    end    
    
end


