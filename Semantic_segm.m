% Picking the folder which contains the data set
path = uigetdir;

% Defining the path of the images/masks
images = path + "\image";
masks = path + "\mask";

% Create the datastores for the trainning sets
classes = ["retinal_blood_vessels","rest_of_the_eye"];
labels = [1 0];
numClasses = 2;



imdsTrain = imageDatastore(images,"ReadFcn",...
            @(x)imresize(rgb2gray(imread(x)),[256,256]));
pxdsTrain =  pixelLabelDatastore(masks,classes,labels,"ReadFcn",@(x)im2bw(imresize(imread(x),[256 256])));

% Defining some usefull parameters for the CNN
tbl = countEachLabel(pxdsTrain);
numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount/numberPixels;
classWeights = 1 ./ frequency;

% Test the dataset
% imgTest = readimage(imdsTrain,1);
% imshow(imgTest)

% Split dataset 80% train-20% validation using my custom function
% splitDataset2 which splits the set into 80-20 for less than 3 inputs,
% otherwise you can put 2 more inputs for (imdsTarin,pxdsTrain,trainRatio,validationRatio)
[ dsTrain, dsVal ] = splitDataSet2( imdsTrain, pxdsTrain);

% Architecture of CNN 
netlayers=[ 
    imageInputLayer([256 256 1]);
%--1d
    convolution2dLayer(3, 32, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer(); % 256 256 32
    maxPooling2dLayer(2, 'Stride', 2); % 128 128 32
%--2d
    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer('Name', 'relu2conv'); % 128 128 128
    maxPooling2dLayer(2, 'Stride', 2); % 64 64 128
%--3d
    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer('Name', 'relu3conv');%<==== concatenate to 3u, 64 64 128
    maxPooling2dLayer(2, 'Stride', 2);% 32 32 128
%--4d
    convolution2dLayer(3, 512, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer(); % 32 32 512
    maxPooling2dLayer(2, 'Stride', 2); % 16 16 512
%------- Expanding path
%--4u
    transposedConv2dLayer([2 2], 512, 'Stride', 2);
    batchNormalizationLayer; 
    reluLayer();% 32 32 512
    
%--3u
    transposedConv2dLayer([2 2], 128, 'Stride', 2);   
    concatenationLayer(3, 2, 'Name', 'concat3');%<========
%    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer();% 64 64 128
%--2u
    transposedConv2dLayer([2 2], 128, 'Stride', 2);
    concatenationLayer(3, 2, 'Name', 'concat2');%<======== 128 128 128
    
    batchNormalizationLayer;
    reluLayer();
%--1u
    transposedConv2dLayer([2 2], 32, 'Stride', 2);
    batchNormalizationLayer;
    reluLayer();

    convolution2dLayer(1, numClasses);

    softmaxLayer();
    pixelClassificationLayer('Classes', classes, 'ClassWeights', classWeights);
];

% Create layer graph
lgraph = layerGraph(netlayers);

% Connect relu3up to the second input of concat
lgraph = connectLayers(lgraph, 'relu3conv', 'concat3/in2');
% Connect relu3up to the second input of concat
lgraph = connectLayers(lgraph, 'relu2conv', 'concat2/in2');
% Analyze the network
analyzeNetwork(lgraph);

netlayers=lgraph;
% Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ValidationData',dsVal,...
    'Plots', 'training-progress');

% netlayers = trainNetwork(dsTrain,netlayers,options);
% save('grayRetinal.mat','netlayers')
load('grayRetinal.mat','netlayers')
imgTestPath = "C:\Users\voulg\OneDrive\Υπολογιστής\MLProject\Retinal_blood_vessels_seg\Data\test\image";
pxdsTestPath = "C:\Users\voulg\OneDrive\Υπολογιστής\MLProject\Retinal_blood_vessels_seg\Data\test\mask";
imdsTest = imageDatastore(imgTestPath,"ReadFcn",@(x)imresize(rgb2gray(imread(x)),[256,256]));
pxdsTest = pixelLabelDatastore(pxdsTestPath,classes,labels,"ReadFcn",@(x)im2bw(imresize(imread(x),[256 256])));

I = readimage(imdsTest,8);
C = semanticseg(I,netlayers,Classes=classes);
B = labeloverlay(I,C);
imshow(B)

dsTest = combine(imdsTest,pxdsTest);
pxdsResults = semanticseg(dsTest,netlayers,...
    Classes=classes,...
    MiniBatchSize=4,...
    WriteLocation=tempdir);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,Verbose=false);




dataSetMetrics = metrics.DataSetMetrics

classMetrics = metrics.ClassMetrics

metrics.ImageMetrics

metrics.ConfusionMatrix

metrics.NormalizedConfusionMatrix
