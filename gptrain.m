function gpmodel=gptrain(X,y,opt)

  addpath /home/mkandemi/Documents/source/code/mil_methods/rgpmil/
   
  N=size(X,1);
  global f;
  global numcalls;
  
  indicestr=opt.likeargs{1};
  ytr=y(indicestr);
  
  finit=ytr*0.001;
  
  f=finit;%  ;
  numcalls=0;
  
  % Is there mean available?
  if length(opt.mean)==1
     m=opt.mean*ones(N,1);
  else
     m=opt.mean;
  end
      
  
  hyp=opt.hyp;
  
  if opt.learnhyp==1                      
      
          for rr=1:5
            %hyp=minimize(hyp, @neglog_mar, -10, y,X,opt);
            [fval,dfval]=neglog_mar(hyp,y,X,opt);
            
            hyp=hyp-0.0000001*dfval;
          end
          
%           hyp.mean=[];
%           hyp.cov=opt.hyp;
%           hyp = minimize(hyp, @gp, -10, @infLaplace, @meanZero, opt.kernfunc, @likLogistic, X, ytr);
%           hyp=hyp.cov;
           
          % learn by grid search
%           hypList=0.1:0.3:7;
%           % hypList=1:3:10;
%           % marList=zeros(length(hypList),1);
%            
%            maxMar=-Inf;
%            bestF=0;
%            bestKinv=0;
%            
%            for ii=1:length(hypList)
%                [lik,K,Kinv,B,W,f]=log_mar(hypList(ii),y,X,opt);
%                
%                %hyp2.cov=hypList(ii);
%                %lik2=gp(hyp2, @infLaplace, @meanZero, @covRBF, @likLogistic, X, y);
%                
%                
%                marList(ii)=lik;
%                %marList2(ii)=-lik2;
%                
%               if lik>maxMar
%                    maxMar=lik;
%                    bestF=f;
%                    hyp=hypList(ii);
%                    bestKinv=Kinv;
%               end
%            end
%            
%            f=bestF;
%            Kinv=bestKinv;
% 
%            figure(1),plot(hypList,marList);
           %figure(2),plot(hypList,marList2);
           
           
           %
                
            
  end              

 
     N=size(X,1);
   
    if isequal(opt.cov,0)
       V=opt.kernfunc(hyp,X)+eye(N)*0.001;
    else
       V=opt.cov;
    end
    Vinv=V\eye(N);        
 
    f=finit;
     f=minimize(finit, @neglog_pos, -opt.maxiter, y,V,Vinv,m,opt);   

     fprintf('hyp: %.2f FINAL acc: %.2f\n',hyp(1),mean(predInstToBag(f,opt.likeargs{1})==(y>0)));


  gpmodel.f=f;
  gpmodel.Kinv=Vinv;
  gpmodel.X=X;
  gpmodel.opt=opt;
  gpmodel.opt.hyp=hyp;
  gpmodel.y=y;
  gpmodel.cov=(-opt.like(f,y,'hess',opt.likeargs)+Vinv+0.001*eye(N))\eye(N)+eye(N);
  
end

% ----------------------------------------------------
%
%  Posterior  
%
% ----------------------------------------------------
function [lik,dg]=neglog_pos(f,y,V,Vinv,m,opt)
   [lik,dg]=log_pos(f,y,V,Vinv,m,opt);
    lik=-lik;
    dg=-dg;    
end

function [lik,dg]=log_pos(f,y,V,Vinv,m,opt)
  N=size(f,1);

   
  lik=opt.like(f,y,'like',opt.likeargs)-0.5*(f-m)'*Vinv*(f-m)-0.5*logdet(V)-0.5*N*log(2*pi);
  
  dg=opt.like(f,y,'grad',opt.likeargs)-Vinv*f+Vinv*m;
end

% ----------------------------------------------------
%
%  Marginal
%
% ----------------------------------------------------
function [lik,dg]=neglog_mar(hyp,y,X,opt)
    
    [lik,K,Kinv,B,W]=log_mar(hyp,y,X,opt);
    lik=-lik;
    
    %dg=-take_gradient(@log_mar,hyp,y,X,opt);
        
   
    dg=-grad_mar(hyp,X,K,Kinv,B,W,opt);
end

function [lik,K,Kinv,B,W,f]=log_mar(hyp,y,X,opt)
    
    global f;
    global numcalls;
    
    
    m=zeros(size(X,1),1);
  
  
    N=size(X,1);
    K=opt.kernfunc(hyp,X)+eye(N)*0.001;
    Kinv=K\eye(N);   
    
   % f=zeros(N,1);      
    
    f=minimize(f, @neglog_pos, -5, y,K,Kinv,m,opt);     
        
   

  W=-max(opt.like(f,y,'hess',opt.likeargs),0);
  B=eye(N)+K*W+eye(N)*0.01;
  B=(B+B')/2;
  
  %B=safecov(B);
  %B= calc_psd_cone(B);
     
  lik=-0.5*f'*Kinv*f-0.5*real(log(det(B)))+opt.like(f,y,'like',opt.likeargs);
  
  
  %fprintf('T1: %.1f T2: %.1f T3: %.1f T4: %.1f\n',-0.5*f'*Kinv*f,-0.5*logdet(K),-0.5*logdet(B),opt.like(f,y,'like',opt.likeargs));
  
  
  numcalls=numcalls+1;
    
  fprintf('.');
  
  if mod(numcalls,25)==0
      fprintf(' %d\n',numcalls);
  end
end

function dg=grad_mar(hyp,X,K,Kinv,B,W,opt)
 global f;
 a=Kinv*f;
 N=length(f);
 
 numHyp=str2double(opt.kernfunc());
 
 W(W<0.001 & W>0)=0.001;

 Winv=W\eye(N);
 Winv(isnan(Winv))=0;
 BBinv=(Winv+K)\eye(N);
 
 dg=zeros(numHyp,1);
 
 for ii=1:numHyp
    dK=opt.kernfunc(hyp,X,X,ii);
    dg(ii)=0.5*a'*dK*a-0.5*trace(BBinv*dK);
 end

end

