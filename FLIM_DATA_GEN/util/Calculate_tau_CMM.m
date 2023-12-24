function tau_CMM = Calculate_tau_CMM (start, stop,h, background_corr, S_start,S_stop, hist)
% The purpose of this function is to calculate the lifetime based on Center-
% of-mass method (CMM)
% Input arguments:
%   start:	start time bin
%   stop:	stop time bin
%   hist:	sum of histogram of decay curve obtained from 192*128 sensor (only
%           quarter of the sensor is used, 96*64)
%   h:      bin width
%   background_corr: Y/N: 1/0
%   S_start:      the start number of bins in the background sampling window
%   S_start:      the stop number of bins in the background sampling window
% Output argument:
%   tau:    lifetime
%
% This function needs the nested function tau_exact
% Write by Dong, 28-2-2019

T = (stop - start + 1)*h;
y = hist;
y_temp = y(start:stop);
n = length(y_temp);
nf = mean(y(S_start:S_stop));
t_temp = (0:n-1)+0.5;

if background_corr == 1
	y_temp = y_temp - nf;
else
	 y_temp = y_temp;
end

tau_CMM = (y_temp*t_temp'/sum(y_temp))*h;
tau = tau_exact(tau_CMM/T)*T;

end