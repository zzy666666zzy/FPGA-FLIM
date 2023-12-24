function y = Generate_decay_sin_exp(tau1,Npeak,FWHM)

%   tau_1,tau_2, alpha: theoretical lifetime components (ns)
%   Npeak:  Peak value of the decay (p.c.)
%   FWHM: FWHM of IRF, (ns)
%   y:  decay curves
%   I:  theoretical IRF

y=zeros(1,256);

t=1:256;   %observation window,suppose 10 time bin width, h=0.04ns
t0=14+randi(3)-2;      %start bin, the one time-bin jitter is taken into consideration
h=0.0390625; %width of time bin
tau1=tau1/h;  % lifetime 1 (ns) 

%IRF  suppose Gaussian function, FWHM=2.35*sig0

sig0 =FWHM/2.3548/h;

%I=exp(-(t-t0).^2/(2.577^2));
I=exp(-(t-t0).^2/(2*sig0^2));

    
%lifetime
 %decay curve
f=exp(-t/tau1); 
% measured decay
    C=conv(I,f);
    y =C(1:length(t));
    y =y /max(y);  
    y=Npeak*y;
    y=round(poissrnd(y));
    y=floor(y);
    y(y<0)=0;
end