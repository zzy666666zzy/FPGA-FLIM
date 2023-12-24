clear
[y,yN,t1map,t2map,fmap]  =generate2D (0.3,2.5,100);
int=sum(y,3);
figure
imagesc(int)
j = jet;
j(1,:) = [ 0 0 0 ];    
colormap(j);
colorbar
axis off
axis equal
set(gca,'fontsize',18)

figure
imagesc(t1map)
colormap(j);
colorbar
axis off
axis equal
caxis([0,3])
set(gca,'fontsize',18)

figure
imagesc(t2map)
colormap(j);
colorbar
axis off
axis equal
caxis([0,3])
set(gca,'fontsize',18)

figure
imagesc(fmap)
colormap(j);
colorbar
axis off
axis equal
caxis([0,1])
set(gca,'fontsize',18)
%%
%LSQ fitting
[y,yN,t1map,t2map,fmap]  =generate2D (0.3,3,100);
FWHM = 0.1673;
t=1:256;
t0=14+randi(3)-2;
h=0.0390625;
sig0 =FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));
x0=[0.5,0.3,2.5];
tic
[t1hat,t2hat,alhat]=LSM_Deconvolution2D(y,I,100,x0);
toc
%%
y_lsq=zeros(256,256,256);
for i=1:256
    for j=1:256
        if i>=28 && i<=228 
           if j>=28 && j<=228
               y_lsq(i,j,:)=Generate_decay_jitter(t1hat(i,j),t2hat(i,j),alhat(i,j),100,FWHM);
           end
        end
    end
end
%%
imagesc(alhat);
h=colorbar;
colormap(jet);
caxis([0, 1]);
set(gca,'looseInset',[0.05 0.05 0.05 0.05])
set(h,'fontsize',16);
set(gca,'FontSize',16);
axis off;
%%
%Test ELM
[y,yN,t1map,t2map,fmap]  =generate2D (0.3,2.5,100);
load ../FLIM_code/FLIM_code/Model/elm_model_BiDecay.mat
start_time_test=cputime;
ResultVec=zeros(256,256,3);
tic
for i=1:256
    for j=1:256
        hist_pp=squeeze(yN(i,j,:)); %use the mormalized histogram
        tempH=hist_pp'*InputWeight'+BiasofHiddenNeurons';
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
toc 
PredictedTau1=ResultVec(:,:,1);
PredictedTau2=ResultVec(:,:,2);
PredictedAlpha=ResultVec(:,:,3);

%%
t=1:256;
t0=14;
I=exp(-(t-t0).^2/(2.577^2));
thresh=100;
y1d=reshape(y,40000,256);
remove_index = any(sum(y1d,2) < thresh, 2);
y1d_reduce=y1d;
y1d_reduce(remove_index,:) = [];
pixelNum = size(y1d,1) - sum(remove_index,1);
[ui, vi, ~, ~] = Calculating_phasor(y1d_reduce, I, 1);
Xedge=[0:0.002:1];
Yedge=[0:0.002:1];
% plot 2D histogram (w/ contour)
figure('Name','Phasor Plot');
h = histogram2(ui,vi,Xedge,Yedge,'DisplayStyle','tile','ShowEmptyBins','on','LineStyle','none');
j = jet;
j(1,:) = [ 1 1 1 ];
colormap(j);
cb=colorbar;
set(cb,'FontSize',16);
ylabel(cb,'phasor counts')
set(gca,'Ydir','Normal')
hold on
% ------ Plot Universal Circle ------ %
x = 0:0.005:1;
circle = sqrt(0.25 - (x - 0.5) .^ 2);
set(gca,'FontSize',18);
set(gca,'looseInset',[0 0 0.1 0])
plot(x,circle,'r','LineWidth',1.5);
axis equal tight
axis([0 1 0 0.6]);

