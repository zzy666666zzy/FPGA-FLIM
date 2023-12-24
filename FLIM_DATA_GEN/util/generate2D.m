function [y,yN,t1map,t2map,fmap] =generate2D (tau1,tau2,Npeak)
%Generate 2D snythetic decay figure
%y:     decay map
%yN:	Normalized decay map
%t1map: ground truth t1
%t2map: ground truth t2
%fmap:  ground truth f
% wirtten by Dong 2020/3/30
data=zeros(256,256,256);
dN=zeros(256,256,256);
t1map=zeros(256,256);
t2map=zeros(256,256);
fmap=zeros(256,256);

for i=1:256
    for j=1:256
        if i>=28 && i<=228
           if j>=28 && j<=228
               data(i,j,:)= Generate_decay_jitter(tau1,tau2,1/200*i-28/200,Npeak);
               t1map(i,j)=tau1;
               t2map(i,j)=tau2;
               fmap(i,j)=1/200*i-28/200;
               dN(i,j,:)=data(i,j,:)./max(data(i,j,:));
           end
        end
    end
end

y=data;
yN=dN;

end
               