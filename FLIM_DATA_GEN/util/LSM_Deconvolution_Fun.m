function [alpha_fit,tau_1,tau_2,y_fit]=LSM_Deconvolution_Fun(y,irf,x0)
% tau:      theoretical lifetime in ns
%input
%t:     time bin
%y:     single decay
%irf:	instrument response function
%x0:	initial guess
  h=0.0390625;
  xdata=1:256;
  y=y(:)';
  x0(2)=x0(2)/h;
  x0(3)=x0(3)/h;
%%
%trust-region-reflective levenberg-marquardt
options = optimset('Display','off','Algorithm','levenberg-marquardt');
fun= @(x,xdata) my_conv2(x,xdata,irf);
yn=y/max(y);
ydata=yn;
[x,~]=lsqcurvefit(fun,x0,xdata,ydata,[],[],options);

y_fit=fun(x,xdata);
alpha_fit=x(1);
tau_1=x(2)*h;
tau_2=x(3)*h;
end