%precision to 2 decimal places
clear
clear
clc
N=40000;
FWHM=0.1673;    %measured FWHM of IRF

label_tau1=zeros(N,1);
label_tau2=zeros(N,1);
label_tau3=zeros(N,1);
label_alpha=zeros(N,1);
label_beta=zeros(N,1);
label_gamma=zeros(N,1);
Npeak=zeros(N,1);
for i =1:N

        tau1 = randi(30)/100 + .1; %0.1 - 0.4
        tau2 = randi(30)/100 + 0.9; %0.9 - 1.2
        tau3 = randi(30)/100 + 1.8; %1.8 - 2.1
        x=randfixedsum(3,1,1,0,1);
        alpha=x(1);
        beta=x(2);
        gamma=x(3);
        Npeak(i) = randi(360) +40; % 40 - 400
        label_tau1(i) = tau1;
        label_tau2(i) = tau2;
        label_tau3(i) = tau3;
        label_alpha(i) = alpha;
        label_beta(i) = beta;
        label_gamma(i) = gamma;
        y(i,:) = Generate_decay_tri_exp_jitter(tau1,tau2,tau3,alpha,beta,gamma,Npeak(i),FWHM);
        y(i,:)=y(i,:)/max(y(i,:));

end
        
    
save('Train_samples_01-04_09_12-18_21-40k-jitter-randfixsum','y','label_tau1','label_tau2','label_tau3','label_alpha','label_beta')