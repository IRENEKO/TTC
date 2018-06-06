%% Generate experimental data for images
clear all;
close all;

% 1. Please put in the name of the image accordingly
picture=imread('.');

% 2. Run the lines
sz=size(picture);
num=prod(sz(1:2));
ratio=0.90;
num=floor(num*ratio); % number of lost entries
ind=randperm(prod(sz(1:2)));
mi=ind(1:num); % the missing pixel
kn=ind(:,num+1:end);

Mi=mi;
Kn=kn;
for i=1:ndims(picture)-1
    Mi=[Mi,i*numel(picture)/3+mi];   % the missing index for all three channel
    Kn=[Kn,i*numel(picture)/3+kn];
end

% 3. Choose your favorite name and name it!
save('favorite.mat','picture','ratio','Mi','Kn','mi','kn')


%% Generate experimental data for videos
clear all;
close all;

% 1. Please put in the name of the video accordingly
video=importdata('.');

% 2. Run the lines
video=permute(video,[1,2,4,3]);
sz=size(video);
num=prod(sz(1:2));
ratio=0.9;
num=floor(num*ratio);   % number of lost entries
ind=randperm(prod(sz(1:2)));
mi=ind(1:num);
kn=ind(:,num+1:end);

Mi=mi;
Kn=kn;
for i=1:prod(sz(3:4))-1
    Mi=[Mi,i*prod(sz(1:2))+mi];   % the missing index for all three channel
    Kn=[Kn,i*prod(sz(1:2))+kn];
end

% 3. Choose your favorite name and name it!
save('favorite.mat','picture','ratio','Mi','Kn','mi','kn')