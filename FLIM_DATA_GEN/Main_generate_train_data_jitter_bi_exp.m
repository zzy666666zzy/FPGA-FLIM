%precision to 2 decimal places
% clear
clc
N=40000;
FWHM=0.1673;    %measured FWHM of IRF
label_tau1=zeros(N,1);
label_tau2=zeros(N,1);
label_alpha=zeros(N,1);
Npeak=zeros(N,1);
parfor i =1:N
        tau1 = (0.5-0.1)*rand(1)+0.1; 
        tau2 = (2).*rand(1)+1; 
        alpha = rand(1);
        Npeak(i) = round(rand(1)*5)+1;
%        Npeak(i) = (360)*rand(1)+40;          
%         Npeak(i) = (250-100).*rand(1) + 100;       %100 ~ 250
%         Npeak(i) = (500-250).*rand(1) + 250;       %250 ~ 500
%         Npeak(i) = (800-500).*rand(1) + 500;       %500 ~ 800
%        Npeak(i) = (2000-100).*rand(1) + 100;          %25 ~ 400
        label_tau1(i) = tau1;
        label_tau2(i) = tau2;
        label_alpha(i) = alpha;
        y(i,:) = Generate_decay_bi_exp(tau1,tau2,alpha,Npeak(i),FWHM);
        yn(i,:)=y(i,:)./max(y(i,:));
end
fprintf('Simulation training data gen done.\n');
save('./TrainingNTestingData/Synthetic_TrainData_bi_decay','y','yn','label_tau1','label_tau2','label_alpha')
%%
%Testing data generation
N=4000;
FWHM=0.1673;    %measured FWHM of IRF

label_tau1=zeros(N,1);
label_tau2=zeros(N,1);
label_alpha=zeros(N,1);
Npeak=zeros(N,1);
parfor i =1:N
        tau1 = (0.5-0.1)*rand(1)+0.1; 
        tau2 = (2).*rand(1)+1; 
        alpha = rand(1);
        Npeak(i) = (360)*rand(1)+40; 
%        Npeak(i) = (100-25).*rand(1) + 25;          %25 ~ 100
%         Npeak(i) = (250-100).*rand(1) + 100;       %100 ~ 250
%         Npeak(i) = (500-250).*rand(1) + 250;       %250 ~ 500
%         Npeak(i) = (800-500).*rand(1) + 500;       %500 ~ 800
%        Npeak(i) = (2000-100).*rand(1) + 100;        
        label_tau1(i) = tau1;
        label_tau2(i) = tau2;
        label_alpha(i) = alpha;
        y(i,:) = Generate_decay_bi_exp(tau1,tau2,alpha,Npeak(i),FWHM);
        yn(i,:)=y(i,:)./max(y(i,:));
end
fprintf('Simulation training data gen done.\n');
save('./TrainingNTestingData/Synthetic_TestData_bi_decay','y','yn','label_tau1','label_tau2','label_alpha');
%%
%ZZY, 24,Mar
%"Train process"
load Synthetic_TrainData_bi_decay.mat
OutputNode=3;  
NumberofHiddenNeurons=500;
ActivF='sigmoid';

train_data=yn;
label=[label_tau1,label_tau2,label_alpha];
TrainingSample=train_data;
NumberofTrainingData=size(TrainingSample,1);
NumberofInputNeurons=size(TrainingSample,2);
NumberofOutputNeurons=OutputNode;

tic
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*TrainingSample';

ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);     
tempH=tempH+BiasMatrix;

switch lower(ActivF)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case {'relu'}
        H = (max(0,tempH));
    case {'sin','sine'}
        H = sin(tempH);    
    case {'hardlim'}
        H = double(hardlim(tempH));
    case {'tribas'}
        H = tribas(tempH);
    case {'radbas'}
        H = radbas(tempH);
end
OutputWeight=pinv(H') * label;
toc
Y=(H' * OutputWeight)';     
MAE=mae(label' - Y);
output=Y;    
fprintf('Training done\n');
fprintf('Training MAE %d,\nof HiddenNeural %d \n',MAE,NumberofHiddenNeurons);
save('./Model/elm_model_BiDecay', 'NumberofInputNeurons', 'NumberofOutputNeurons', 'InputWeight',...
    'BiasofHiddenNeurons', 'OutputWeight', 'ActivF', 'label');
%%
%ELM synthetic test data prediction
load Synthetic_TestData_bi_decay.mat
load elm_model_BiDecay.mat

NumberofTestingData=size(yn,1);
tempH_test=InputWeight*yn';           
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);  %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivF)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'relu'}
        H_test = (max(0,tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}
        H_test = hardlim(tempH_test);        
            
end
ResultVec=H_test'*OutputWeight;                          
PredictedTau1=ResultVec(:,1);
PredictedTau2=ResultVec(:,2);
PredictedAlpha=ResultVec(:,3);
label=[label_tau1,label_tau2,label_alpha];
TestingAccuracy=mae(label - ResultVec);
fprintf('Testing MAE %d\n',TestingAccuracy);
%%
%LMQ synthetic test
load Synthetic_TestData_bi_decay.mat
t=1:256;
t0=14+randi(3)-2;
h=0.0390625;
FWHM=0.1673;
sig0 =FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));
% Trust-Region-Reflective Levenberg-Marquardt
x0=[0.5,0.5,2];
tic
[tau1_lsq,tau2_lsq,alpha_lsq]=LSM_Deconvolution(y,I,x0);
toc
fprintf('LSQ fitting done\n');
%%
%LSQ real-world sample fitting 
%load PC3_256256.mat
%load hek256256.mat
load pc3probe1.mat
%load cfp_yfp_2_19July2019.mat
%hist256256=reshape(hist,256,256,256);
t=1:256;
FWHM = 0.1673;
t0=14+randi(3)-2;
h=0.0390625;
sig0 =FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));
x0=[0.1,0.1,0.5];
threshold=100;
tic
[tau1_lsq,tau2_lsq,alpha_lsq]=LSM_Deconvolution2D(hist256256,I,threshold,x0);
toc
tau_ave_lsq=tau1_lsq.*alpha_lsq+(1-alpha_lsq).*tau2_lsq;
imagesc(tau_ave_lsq);
h=colorbar;
colormap(jet);
caxis([0, 2]);
%%
%Real-world data ELM predict process
load elm_model_BiDecay.mat
%load PC3_probe4.mat
load pc3probe1.mat
%load hek256256.mat
%load cfp_yfp_2_19July2019.mat
hist256256=reshape(hist,256,256,256);
threshold=100;
hist_intensity=sum(hist256256,3);
tau_ave=zeros(256,256);
tic
ResultVec=zeros(256,256,3);
for i=1:256
    for j=1:256   
        if hist_intensity(i,j)<threshold
            ResultVec(i,j,:)=0;
        else    
            hist_pp=squeeze(hist256256(i,j,:));
            hist_pp=hist_pp/max(hist_pp);
            tempH=hist_pp'*InputWeight';+BiasofHiddenNeurons';
            switch lower(ActivF)
                case {'sig','sigmoid'}
                    H = 1 ./ (1 + exp(-tempH));
                case {'sin','sine'}
                    H = sin(tempH);        
                case {'hardlim'}
                    H = hardlim(tempH);                
            end 
            ResultVec(i,j,:)=H*OutputWeight;
        end
    end
end
toc
PredictedTau1=ResultVec(:,:,1);
PredictedTau2=ResultVec(:,:,2);
PredictedAlpha=ResultVec(:,:,3);
tau_ave_elm=PredictedTau1.*PredictedAlpha+(1-PredictedAlpha).*PredictedTau2;
imagesc(tau_ave_elm);
h=colorbar;
colormap(jet);
caxis([0, 2]);
fprintf('Prediction Done\n');
%%
%ELM boxplot
% ae_tau1_elm_25_100=abs(label_tau1-PredictedTau1);
% ae_tau2_elm_25_100=abs(label_tau2-PredictedTau2);
% ae_alpha_elm_25_100=abs(label_alpha-PredictedAlpha);

% ae_tau1_elm_100_250=abs(label_tau1-PredictedTau1);
% ae_tau2_elm_100_250=abs(label_tau2-PredictedTau2);
% ae_alpha_elm_100_250=abs(label_alpha-PredictedAlpha);
% 
% ae_tau1_elm_250_500=abs(label_tau1-PredictedTau1);
% ae_tau2_elm_250_500=abs(label_tau2-PredictedTau2);
% ae_alpha_elm_250_500=abs(label_alpha-PredictedAlpha);
% 
% ae_tau1_elm_500_800=abs(label_tau1-PredictedTau1);
% ae_tau2_elm_500_800=abs(label_tau2-PredictedTau2);
% ae_alpha_elm_500_800=abs(label_alpha-PredictedAlpha);
%save('./boxplot_data/elm_25_100','ae_tau1_elm_25_100','ae_tau2_elm_25_100','ae_alpha_elm_25_100');
%save('./boxplot_data/elm_100_250','ae_tau1_elm_100_250','ae_tau2_elm_100_250','ae_alpha_elm_100_250');
%save('./boxplot_data/elm_250_500','ae_tau1_elm_250_500','ae_tau2_elm_250_500','ae_alpha_elm_250_500');
%save('./boxplot_data/elm_500_800','ae_tau1_elm_500_800','ae_tau2_elm_500_800','ae_alpha_elm_500_800');
% 
load elm_25_100.mat
load elm_100_250.mat
load elm_250_500.mat
load elm_500_800.mat

X1 = [1 3 5 7];
X2 = [1.4 3.4 5.4 7.4];
X3 = [1.8 3.8 5.8 7.8];

data_tau1=[ae_tau1_elm_25_100,ae_tau1_elm_100_250,ae_tau1_elm_250_500,ae_tau1_elm_500_800];
data_tau2=[ae_tau2_elm_25_100,ae_tau2_elm_100_250,ae_tau2_elm_250_500,ae_tau2_elm_500_800];
data_alpha=[ae_alpha_elm_25_100,ae_alpha_elm_100_250,ae_alpha_elm_250_500,ae_alpha_elm_500_800];

boxplot(data_tau1,'position',X1,'symbol','','widths',0.25,'colors','b');
hold on
boxplot(data_tau2,'position',X2,'symbol','','widths',0.25,'colors','r');
hold on
boxplot(data_alpha,'position',X3,'symbol','','widths',0.25,'colors','g');
set(gca,'FontSize',14);
set(gca,'looseInset',[0.05 0.05 0.05 0.05])
xlabel('Peak intensity','FontSize',14)
ylabel('Absolute error','FontSize',14)
xlim([0 8.5])
ylim([0 0.6])
%%
%LSQ boxplot
% ae_tau1_lsq_25_100=abs(label_tau1-tau1_lsq);
% ae_tau2_lsq_25_100=abs(label_tau2-tau2_lsq);
% ae_alpha_lsq_25_100=abs(label_alpha-alpha_lsq);

% ae_tau1_lsq_100_250=abs(label_tau1-tau1_lsq);
% ae_tau2_lsq_100_250=abs(label_tau2-tau2_lsq);
% ae_alpha_lsq_100_250=abs(label_alpha-alpha_lsq);
% 
% ae_tau1_lsq_250_500=abs(label_tau1-tau1_lsq);
% ae_tau2_lsq_250_500=abs(label_tau2-tau2_lsq);
% ae_alpha_lsq_250_500=abs(label_alpha-alpha_lsq);
% 
% ae_tau1_lsq_500_800=abs(label_tau1-tau1_lsq);
% ae_tau2_lsq_500_800=abs(label_tau2-tau2_lsq);
% ae_alpha_lsq_500_800=abs(label_alpha-alpha_lsq);

%save('./boxplot_data/lsq_25_100','ae_tau1_lsq_25_100','ae_tau2_lsq_25_100','ae_alpha_lsq_25_100');
%save('./boxplot_data/lsq_100_250','ae_tau1_lsq_100_250','ae_tau2_lsq_100_250','ae_alpha_lsq_100_250');
%save('./boxplot_data/lsq_250_500','ae_tau1_lsq_250_500','ae_tau2_lsq_250_500','ae_alpha_lsq_250_500');
%save('./boxplot_data/lsq_500_800','ae_tau1_lsq_500_800','ae_tau2_lsq_500_800','ae_alpha_lsq_500_800');

load lsq_25_100.mat
load lsq_100_250.mat
load lsq_250_500.mat
load lsq_500_800.mat

X1 = [1 3 5 7];
X2 = [1.4 3.4 5.4 7.4];
X3 = [1.8 3.8 5.8 7.8];

data_tau1=[ae_tau1_lsq_25_100,ae_tau1_lsq_100_250,ae_tau1_lsq_250_500,ae_tau1_lsq_500_800];
data_tau2=[ae_tau2_lsq_25_100,ae_tau2_lsq_100_250,ae_tau2_lsq_250_500,ae_tau2_lsq_500_800];
data_alpha=[ae_alpha_lsq_25_100,ae_alpha_lsq_100_250,ae_alpha_lsq_250_500,ae_alpha_lsq_500_800];

boxplot(data_tau1,'position',X1,'symbol','','widths',0.25,'colors','b');
hold on
boxplot(data_tau2,'position',X2,'symbol','','widths',0.25,'colors','r');
hold on
boxplot(data_alpha,'position',X3,'symbol','','widths',0.25,'colors','g');
set(gca,'FontSize',14);
set(gca,'looseInset',[0.05 0.05 0.05 0.05])
xlabel('Peak intensity','FontSize',14)
ylabel('Absolute error','FontSize',14)
xlim([0 8.5])
ylim([0 1.4])




