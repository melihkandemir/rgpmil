function [ynew,fnew]=gppredict(Xts,gpmodel)

  ktest=gpmodel.opt.kernfunc(gpmodel.opt.hyp,Xts,gpmodel.X);
  
  fnew=ktest*gpmodel.Kinv*gpmodel.f;
  
  fnew(fnew==0)=0.001;
  
  ynew=sign(fnew);
  
end