%% CNN for image classification
%+++++++++++++++++++++++++++++++++++

clear all
close all
clc

%% PARAMETERS
% the name of the file where the trained CNN is saved
nameFileRez='rezCNN2.mat';

% the size of the input images
inputSize = [227 227 3]; 

% training parameters
MBS=32;% mini batch size
NEP=5; % number of epochs

%% TRAINING AND VALIDATION DATASETS

% indicate the path to the trainingaand validation images
pathImagesTrain='/Users/silvia-teodoraporcarasu/Documents/ML/archive';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize

% create the datastore with the training and validation images
imds = imageDatastore(pathImagesTrain, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

resizeImagesImds(imds,'/Users/silvia-teodoraporcarasu/Documents/ML/archive/train',inputSize(1:2));

% split the dataset into training and validation datasets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images 
numClasses = numel(categories(imdsTrain.Labels)); %the number of classes

% augment the training and validation dataset
pixelRange = [-30 30];   
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% TESTING DATASETS

% indicate the path to the testing dataset
pathImagesTest='/Users/silvia-teodoraporcarasu/Documents/ML/archive';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize

% create the datastore for the testing dataset      
imdsTest = imageDatastore(pathImagesTest, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');     

resizeImagesImds(imdsTest, '/Users/silvia-teodoraporcarasu/Documents/ML/archive/test', inputSize(1:2));

%% DESIGN THE ARCHITECTURE

% load the pretrained model 
net = alexnet; 

% take the layers for transfer of learning
layersTransfer = net.Layers(1:end-3); 

% create the new architecture: last fully connected layer for the required number of classes
layersNew = [
    layersTransfer    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];

%% DESIGN THE ARCHITECTURE

% load the pretrained model 
net = alexnet; 

% take the kayers for transfer of learning
layersTransfer = net.Layers(1:end-3); 

% create the new architecture: the last fully connected layer is configured for the necessary number of classes
layersNew = [
    layersTransfer    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];

%% TRAIN THE CNN

% indicate the training parameters
options = trainingOptions('adam', ...
    'MiniBatchSize',MBS,...            
    'MaxEpochs',NEP, ...      
    'InitialLearnRate',1e-4, ...  
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
                  
% train the model
netTransfer = trainNetwork(augimdsTrain,layersNew,options);

% save the trained model
feval(@save,nameFileRez,'netTransfer'); 

%% VERIFY THE RESULTS

% Add Gaussian noise
% gaussianNoise = 0.5; 
% addGaussianNoise = @(x) imnoise(x, 'gaussian', 0, gaussianNoise);
% imdsTestNoisy = transform(imdsTest, addGaussianNoise);

% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer,imdsValidation); 
accuracyValidation = mean(YPredValidation == imdsValidation.Labels)  

% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer,imdsTrain);  
accuracyTrain = mean(YPredTrain == imdsTrain.Labels)  

% testing - responses and accuracy after adding noise
[YPredTestNoisy,scoresTestNoisy] = classify(netTransfer, imdsTestNoisy);  
accuracyTestNoisy = mean(YPredTestNoisy == imdsTest.Labels)  
