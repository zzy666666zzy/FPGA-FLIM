%%
%Training data generation
N=10000;
FWHM=0.1673;    %measured FWHM of IRF
label_tau1=zeros(N,1);
Npeak=zeros(N,1);
parfor i =1:N
        tau1 = (5-0.1).*rand(1) + 0.1;
%        Npeak(i) = round(rand(1)*5)+1;   
        Npeak(i) = (360)*rand(1)+40;
%         Npeak(i) = (250-100).*rand(1) + 100;       %100 ~ 250
%         Npeak(i) = (500-250).*rand(1) + 250;       %250 ~ 500
%         Npeak(i) = (800-500).*rand(1) + 500;       %500 ~ 800
        label_tau1(i) = tau1;
        y(i,:) = Generate_decay_sin_exp(tau1,Npeak(i),FWHM);
        yn(i,:)=y(i,:)/max(y(i,:));
end
fprintf('Generate training data done\n');
save('./TrainingNTestingData/Synthetic_TrainData_single_decay','y','yn','label_tau1')
%%
%Testing data generation
N=1000;
FWHM=0.1673;    %measured FWHM of IRF

label_tau1=zeros(N,1);
Npeak=zeros(N,1);
for i =1:N
        tau1 = (5-0.1).*rand(1) + 0.1; %0.1 ~ 5
%        Npeak(i) = (100-25).*rand(1) + 25;          %25 ~ 100
%         Npeak(i) = (250-100).*rand(1) + 100;       %100 ~ 250
%         Npeak(i) = (500-250).*rand(1) + 250;       %250 ~ 500
         Npeak(i) = (800-500).*rand(1) + 500;       %500 ~ 800
        label_tau1(i) = tau1;
        y(i,:) = Generate_decay_sin_exp(tau1,Npeak(i),FWHM);
        yn(i,:)=y(i,:)/max(y(i,:));
end
fprintf('Generate testing data done\n');
save('./TrainingNTestingData/Synthetic_TestData_single_decay','y','yn','label_tau1')
%%
%ZZY, 23,Mar
%ELM Train process
load Synthetic_TrainData_single_decay.mat
OutputNode=1;  
NumberofHiddenNeurons=800;
ActivF='sig';

train_data=yn;
label=label_tau1;
TrainingSample=train_data;
NumberofTrainingData=size(TrainingSample,1);
NumberofInputNeurons=size(TrainingSample,2);
NumberofOutputNeurons=OutputNode;

tic
%Randomly set the weights between input and hidden layer
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
%Randomly set bias values
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
%Map input to feature map
tempH=InputWeight*TrainingSample';

ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);     
tempH=tempH+BiasMatrix;

switch lower(ActivF)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
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
%actual output value of trained data
Y=(H' * OutputWeight)';     
MAE=mae(label' - Y);
output=Y;    
fprintf('Training done\n');
fprintf('Training MAE %d,\nof HiddenNeural %d \n',MAE,NumberofHiddenNeurons);
save('./Model/elm_model_SingleDecay', 'NumberofInputNeurons', 'NumberofOutputNeurons', 'InputWeight',...
    'BiasofHiddenNeurons', 'OutputWeight', 'ActivF', 'label');
%%
%ELM synthetic test data prediction elm
load Synthetic_TestData_single_decay.mat
load elm_model_SingleDecay.mat

NumberofTestingData=size(yn,1);

%%%%%%%%%%% Calculate the output of testing input
tic
tempH_test=InputWeight*yn';
clear y;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind); %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivF)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}
        H_test = hardlim(tempH_test);               
end
Y=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
toc
TestingAccuracy=mae(label_tau1' - Y);
fprintf('ELM prediction accuracy %d\n',TestingAccuracy);
%%
%LSQ synthetic test data fitting 
load Synthetic_TestData_single_decay.mat
t=1:256;
FWHM = 0.1673;
t0=14+randi(3)-2;
h=0.0390625;
sig0 =FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));
fun=@(tau1,t) convo(tau1,t,I);
tic
options = optimset('Display','off','Algorithm','levenberg-marquardt');
for i=1:size(y,1)
    hist_pp=yn(i,:);
    [tau_hat(i,:),~] = lsqcurvefit(fun, 2/h, t, hist_pp,[],[],options);
    fprintf('%d\n',i);
end
tau_hat=tau_hat*h;
toc
FittingAccuracy=mae(label_tau1 - tau_hat);
fprintf('LSQ fitting accuracy %d\n',FittingAccuracy);
%%
%LSQ real-world sample fitting 
load hist_tsa201_cfp_3June2019.mat
load irf1.mat
I=zeros(1,256);
I=irf/max(irf);
threshold=200;
x0=5;
tic
[tau]=LSM_Deconvolution2D_sin(hist,I,threshold,x0);
toc
tau_lsq=tau;
fprintf('LSQ prediction Done\n');
%%
%ELM real-world sample prediction process
load elm_model_SingleDecay.mat
load hist_tsa201_cfp_3June2019.mat
%filter out the background, threshold 100
threshold=200;
hist256256=reshape(hist,256,256,256);
hist_intensity=sum(hist256256,3);
tic
for i=1:256
    for j=1:256
        if hist_intensity(i,j)<threshold
            tau(i,j)=0;
        else
            hist_pp=squeeze(hist256256(i,j,:));
            hist_pp=hist_pp/max(hist_pp);
            tempH=hist_pp'*InputWeight'+BiasofHiddenNeurons';
            switch lower(ActivF)
                case {'sig','sigmoid'}
                    H = 1 ./ (1 + exp(-tempH));
                case {'sin','sine'}
                    H = sin(tempH);        
                case {'hardlim'}
                    H = hardlim(tempH);                
            end    
            tau(i,j)=H*OutputWeight;
       end
    end
end
toc
tau_elm=tau;
fprintf('ELM prediction Done\n');
%%
%ELM boxplot
% ae_tau1_elm_25_100=abs(label_tau1-Y');
% %ae_tau1_elm_100_250=abs(label_tau1-Y');
% %ae_tau1_elm_250_500=abs(label_tau1-Y');
% %ae_tau1_elm_500_800=abs(label_tau1-Y');
% 
% save('./boxplot_data/monodecay_elm_25_100','ae_tau1_elm_25_100');
% %save('./boxplot_data/monodecay_elm_100_250','ae_tau1_elm_100_250');
% %save('./boxplot_data/monodecay_elm_250_500','ae_tau1_elm_250_500');
% %save('./boxplot_data/monodecay_elm_500_800','ae_tau1_elm_500_800');

load monodecay_elm_25_100.mat
load monodecay_elm_100_250.mat
load monodecay_elm_250_500.mat
load monodecay_elm_500_800.mat
% 
X1 = [1 3 5 7];

data_tau1=[ae_tau1_elm_25_100,ae_tau1_elm_100_250,ae_tau1_elm_250_500,ae_tau1_elm_500_800];

boxplot(data_tau1,'position',X1,'symbol','','widths',0.5);
set(gca,'FontSize',14);
set(gca,'looseInset',[0.05 0.05 0.05 0.05])
xlabel('Peak intensity','FontSize',14)
ylabel('Absolute error','FontSize',14)
xlim([0 8])
ylim([0 0.18])
%%
%LSQ boxplot
%boxplot
%ae_tau1_lsq_25_100=abs(label_tau1-tau_hat);
%ae_tau1_lsq_100_250=abs(label_tau1-tau_hat);
%ae_tau1_lsq_250_500=abs(label_tau1-tau_hat);
%ae_tau1_lsq_500_800=abs(label_tau1-tau_hat);
% 
%save('./boxplot_data/monodecay_lsq_25_100','ae_tau1_lsq_25_100');
% %save('./boxplot_data/monodecay_lsq_100_250','ae_tau1_lsq_100_250');
% %save('./boxplot_data/monodecay_lsq_250_500','ae_tau1_lsq_250_500');
% save('./boxplot_data/monodecay_lsq_500_800','ae_tau1_lsq_500_800');

load monodecay_lsq_25_100.mat
load monodecay_lsq_100_250.mat
load monodecay_lsq_250_500.mat
load monodecay_lsq_500_800.mat

X1 = [1 3 5 7];

data_tau1=[ae_tau1_lsq_25_100,ae_tau1_lsq_100_250,ae_tau1_lsq_250_500,ae_tau1_lsq_500_800];

boxplot(data_tau1,'position',X1,'symbol','','widths',0.5);
set(gca,'FontSize',14);
set(gca,'looseInset',[0.05 0.05 0.05 0.05])
xlabel('Peak intensity','FontSize',14)
ylabel('Absolute error','FontSize',14)
xlim([0 8])
ylim([0 1.6])