%% Experiment 1
clear all
close all
cd data
load 1.mat

% Tunning parameters
para.h=22;                  % scaling factor h 
para.maxitr=2;    
para.lambda=[0.2,0.2];      % (For TV term) 
para.R2=6;
para.R=11;
para.Rd=17;
para.n=[9 6 6 6 9 9 3];   
para.idf=[1,4,7];           % (For TV term) the identifier of the start of original tensor dimensions

para.mi=mi;
para.kn=kn;
para.Mi=mi;
para.Kn=Kn;

version=1;   % version=1: TTC; version=2: TTV-TV
simpic=repro_exp1(picture,para,version);


