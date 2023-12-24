clear;
numbin_128 = 128;
Q=1.00992;%floor((q^100 - 1) / (q-1)  = 256

numbin_80 = 80;
Q1=1.0256;%floor((q^80 - 1) / (q+1)  = 256

bin_128=0;%Bin number select 128 or 100
if bin_128==1
    numbin=numbin_128;
    q=Q;
else
    numbin=numbin_80;
    q=Q1;
end

bin_idx = 1:1:numbin;
load PC_100_cycle;

% load 'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave\Synthetic_2D_Pattern_1000to5000.mat'
% hist=decay;

up = floor((power(q, bin_idx) - 1) / (q-1))+1;
low = floor((power(q, bin_idx - 1) - 1) /(q-1))+1;

coeffi=0.012;
spatial_resolution=65536;

% m_up=floor((1-exp(-bin_idx*coeffi))*256);
% m_low=floor((1-exp(-(bin_idx-1)*coeffi))*256);

log_rates = zeros(spatial_resolution,numbin);
ori_log_rates=zeros(spatial_resolution,numbin);
up(80)=up(80)-1;
for ii = 1:numbin %range should be ajusted
    log_rates(:,ii)= sum(hist(:,low(ii):up(ii)), 2);
    ori_interval=low(ii):up(ii);
end

ThreeD=reshape(log_rates,256,256,80);
inten=sum(ThreeD,3);
hist_total_1d=sum(log_rates,1);

% for ii = 1:numbin %range should be ajusted
%     if 1<ii && ii<14
%         log_rates(:,ii) = sum(hist(:,m_low(ii):m_up(ii)), 2);%Binning operaion
%         interval=m_low(ii):m_up(ii);
%     elseif 14<=ii && ii<=numbin
%         log_rates(:,ii)= sum(hist(:,low(ii):up(ii)), 2);
%         ori_interval=low(ii):up(ii);
%     end
% end
%%
log_rates=[log_rates(:,1:14),log_rates(:,14:numbin)];
log_rates_sqz=sum(log_rates,1);
log_rates_cube=reshape(log_rates,608,736,77);

for ii = 2:numbin
    ori_log_rates(:,ii)= sum(spad(:,low(ii):up(ii)), 2);
end
ori_log_rates_sqz=sum(ori_log_rates,1);
ori_log_rates_cube=reshape(ori_log_rates,608,736,128);
