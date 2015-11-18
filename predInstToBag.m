function [ybag,fbag]=predInstToBag(finst,indices)
   
    B=max(indices);
    fbag=zeros(B,1);
    ybag=zeros(B,1);

    for bb=1:B
       %fbag(bb)=1/(1+1/sum(exp(finst(indices==bb))))-0.5;
       fbag(bb)=max(finst(indices==bb));
       ybag(bb)=sign(fbag(bb));
    end
   
end
