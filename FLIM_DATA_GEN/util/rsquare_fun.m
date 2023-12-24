function [r2] =rsquare_fun (target,predict)
r2 = 1 - sum((target(:)-predict(:)).^2)/sum((target(:)).^2);
end