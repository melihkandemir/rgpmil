clear;
load /home/mkandemi/Documents/data/nokia/nokia_long_equal.mat
addpath(genpath('/home/mkandemi/Dropbox/code/gpml'))
addpath(genpath('/home/mkandemi/Dropbox/code/mil_methods/rgpmil'))
addpath /home/mkandemi/Dropbox/code/util/

load /home/mkandemi/Dropbox/data/iris.data


if 1==1
    [N,D]=size(iris);
    y=iris(:,D);
    X=zscore(iris(:,1:(D-1)));
    D=D-1;
    ybin= 2*(y==1)-1;
else
    [N,D]=size(X);
    X=zscore(X);
    ybin=ybin(:,3);
end


meanfunc = @meanZero;      hyp.cov=5;
covfunc = @covRBF;       likfunc = @likLogistic;


for ii=1:N
    data(ii).instance=X(ii,:);
    data(ii).label=ybin(ii);
    data(ii).inst_label=ybin(ii);
end

for rep=1:10
    rep
    idx=randperm(N);
    Ntr=floor(N*0.75);
    traindata=data(idx(1:Ntr));
    testdata=data(idx(Ntr+1:end));
    Xtr=X(idx(1:Ntr),:);
    Xts=X(idx(Ntr+1:end),:);
    ytr=ybin(idx(1:Ntr));
    yts=ybin(idx(Ntr+1:end));

    % Rasmussen's code

    hyp2 = minimize(hyp, @gp, -20, @infLaplace, meanfunc, covfunc, likfunc, Xtr, ytr);
    
    [~,~,fpred,~,~]= gp(hyp2, @infLaplace, meanfunc, covfunc, likfunc, Xtr, ytr,Xts,ones(length(yts),1));
    [~,~,ftr,~,~]= gp(hyp2, @infLaplace, meanfunc, covfunc, likfunc, Xtr, ytr,Xtr,ones(length(ytr),1));
        
    mean(ytr==sign(ftr))
    
    fpred(fpred==0)=0.001;
    
    acc1(rep)=mean(yts==sign(fpred));

    % My code

    opt.hyp=hyp.cov;               
    opt.mean=0;
    opt.cov=0;
    opt.kernfunc=@covRBF;
    opt.learnhyp=1;
    opt.like=@likeBinMILClass;
    opt.likeargs{1}=1:length(ytr); %indices

    
    gpmodel=gptrain(Xtr,ytr,opt);
    ypred2=gppredict(Xts,gpmodel);
    
    acc2(rep)=mean(yts==ypred2);
end
