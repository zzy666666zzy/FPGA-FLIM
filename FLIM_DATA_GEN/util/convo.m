function F = convo(x,xdata,I)
% convolution with bi-exponential decay

f=exp(-xdata/x);
F=conv(I,f);
F=F/max(F);
F=F(1:length(xdata));
end