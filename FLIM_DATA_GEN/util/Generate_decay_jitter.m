function [y, y_t] = Generate_decay_jitter(tau1,tau2,alpha,Npeak,FWHM)
%   tau_1,tau_2, alpha: theoretical lifetime components (ns)
%   Npeak:  Peak value of the decay (p.c.)
%   FWHM: FWHM of IRF, (ns)
%   y:  decay curves
%   y_t: theoreticl decay
%   I:  theoretical IRF
if nargin <=4
    FWHM = 0.1673;
end
y=zeros(1,256);

t=1:256;   %observation window,suppose 10 time bin width, h=0.04ns
t0=14+randi(3)-2;      %start bin, the one time-bin jitter is taken into consideration
h=0.0390625; %width of time bin
tau1=tau1/h;  %shorter lifetime (ns)
tau2=tau2/h;  %longer lifetime (ns)  

%IRF  suppose Gaussian function, FWHM=2.35*sig0

sig0 =FWHM/2.3548/h;
%I=exp(-(t-t0).^2/(2.577^2));
I=exp(-(t-t0).^2/(2*sig0^2));

    
%lifetime
 %decay curve
f=alpha*exp(-t/tau1)+(1-alpha)*exp(-t/tau2); 
% measured decay
    C=conv(I,f);
    y =C(1:length(t));
    y =y /max(y);  
    y=Npeak*y;
    y_t = y;
%add Poisson noise
    y =y + sqrt(y).*randn(1,length(t));
    y=floor(y);
    y(y<0)=0;
   
    
end