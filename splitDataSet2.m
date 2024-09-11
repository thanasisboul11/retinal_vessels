function [dsTrain,dsVal] = splitDataSet2(imdsTrain,pxdsTrain,trainRatio,validationRatio)
    if nargin < 3
        trainRatio = 0.8;
        validationRatio = 0.2;
    end
    
    numFiles = numel(imdsTrain.Files);
    indices = randperm(numFiles);

    numTrain = round( trainRatio * numFiles );
    numVal = round( validationRatio * numFiles );

    trainIndices = indices( 1 : numTrain );
    valIndices = indices( numTrain+1 : end );

    imdsTrainSplit = subset( imdsTrain, trainIndices );
    imdsValSplit = subset( imdsTrain, valIndices );

    pxdsTrainSplit = subset( pxdsTrain, trainIndices );
    pxdsValSplit = subset( pxdsTrain, valIndices );

    dsTrain = combine(imdsTrainSplit,pxdsTrainSplit);
    dsVal = combine(imdsValSplit,pxdsValSplit);

end