function [ui, vi, uo, vo] = Calculating_phasor(hist, irfa, threshold)
% Calculate phasor coordinate
% Input
% hist: the histogram of 2d FLIM data
% irfa: the IRF
% threshold: intensity threshold, if the inetnsity is lower than the thres, the lifetime is 0.
% Output:
% uo, io: uncalibrated phasor coordinate
% ui, vi: calibrated phasor coordinate by eliminating the effect of IRF 
% irf and hist should be aligned by irf_align beforehand

sh = size(hist);
si = size(irfa);

if (sh(2) ~= si(2))
    error('No. of Hist and IRF time bins are unequal');
else
    T = sh(2);
end

if (si(1) == 1)
    pixel_no = sh(1);
elseif (si(1) ~= sh(1))
    error('Image size of Hist and IRF are unequal');
else
    pixel_no = sh(1);
end

t = 0.5:1:T-0.5;
cs = cos(2*pi*t/T);
ss = sin(2*pi*t/T);

[ui, vi, uo, vo] = deal(zeros(sh(1),1));
for i = 1:pixel_no
    y = hist(i,:);
    if (si(1) == 1)
        irf = irfa;
    else
        irf = irfa(i,:);
    end
    intensity = sum(y);
    if (intensity < threshold)
        uo(i) = 0.0;
        vo(i) = 0.0;
        ui(i) = 0.0;
        vi(i) = 0.0;
    else
        uo(i) = y*cs'/sum(y);
        vo(i) = y*ss'/sum(y);
        u_i = irf*cs'/sum(irf);
        v_i = irf*ss'/sum(irf);
        ui(i) = uo(i)*u_i + vo(i)*v_i;
        vi(i) = -uo(i)*v_i + vo(i)*u_i;
    end
end


