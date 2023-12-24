function F = my_conv2(x,xdata,I)
% convolution with bi-exponential decay

f=x(1)*exp(-xdata/x(2))+(1-x(1))*exp(-xdata/x(3));
F=conv(I,f);
F=F/max(F);
F=F(1:length(xdata));

end