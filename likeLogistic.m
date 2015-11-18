function res=likeLogistic(f, y, str,varargin)
  % log of logistic likelihood
  
  if strcmp(str,'like')==1
    %  yf = y.*f; s = -yf;                   % product latents and labels
    %  ps   = max(0,s); 
    %  res = sum(-(ps+log(exp(-ps)+exp(s-ps))));                % lp = -(log(1+exp(s)))
      
      res=sum(-log(1+exp(-f.*y)));  
  elseif strcmp(str,'grad')==1
      res=gradLogistic(f,y);
  elseif strcmp(str,'hess')==1
      res=hessLogistic(f);
      res2=-take_hessian(@gradLogistic,f,y);
      aa=0;
  end
end

function dg=gradLogistic(f,y)
    t=(y+1)/2;
    pp=logistic_func(f);
    dg=t-pp;

end

function W=hessLogistic(f)
   pp=logistic_func(f);
   
   W=-diag(-pp.*(1-pp));    
end


