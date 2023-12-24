function [tau]=LSM_Deconvolution_Fun_sin(y,irf,x)
% tau:      theoretical lifetime in ns
%input
%t:     time bin
%y:     single decay
%irf:	instrument response function
%x0:	initial guess
  h=0.0390625;
  xdata=1:256;
  y=y(:)';
  x=x/h;
%%
%trust-region-reflective levenberg-marquardt
options = optimset('Display','off','Algorithm','levenberg-marquardt');
fun= @(x,xdata) convo(x,xdata,irf);
yn=y/max(y);
ydata=yn;
[x]=lsqcurvefit(fun,x,xdata,ydata,[],[],options);

y_fit=fun(x,xdata);
tau=x*h;
end