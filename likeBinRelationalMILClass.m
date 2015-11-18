function res=likeBinRelationalMILClass(f, y, str, varargin)
  R=varargin{1}{2}; 
  
  res=likeBinMILClass(f,y,str,varargin{1})+likeLink(f,R,str,varargin{1});

end