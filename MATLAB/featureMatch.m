% Aaron Thomas
% Prints feature matching across images.
% tested on Matlab r2014a with Computer Vision System Toolbox
% solution is based on RANSAC with randomness
% every time you run this code, it may produce slightly different results

%% Read images
colorA = imread('seoul1.jpg');
colorB = imread('seoul2.jpg');

figure;
subplot(1,2,1); imshow(colorA); title('Input image A');
subplot(1,2,2); imshow(colorB); title('Input image B');

%% Convert color image to gray scale for feature extraction
grayA = rgb2gray(colorA);
grayB = rgb2gray(colorB);

%% Detect keypoints
pointsA = detectSURFFeatures(grayA);
pointsB = detectSURFFeatures(grayB);

%% Extract descriptors on each keypoint
[featuresA, validPointsA] = extractFeatures(grayA, pointsA);
[featuresB, validPointsB] = extractFeatures(grayB, pointsB);

figure;
subplot(1,2,1); imshow(grayA); hold on; plot(validPointsA,'showOrientation',true); title('Extracted SURF keypoints');
subplot(1,2,2); imshow(grayB); hold on; plot(validPointsB,'showOrientation',true); title('Extracted SURF keypoints');

%% Match features
indexPairs = matchFeatures(featuresA, featuresB);
matchedPointsA = validPointsA(indexPairs(:, 1), :);
matchedPointsB = validPointsB(indexPairs(:, 2), :);

figure; 
showMatchedFeatures(grayA,grayB,matchedPointsA,matchedPointsB,'montage');
title('Matched SURF points, including outliers');

%% Use RANSAC to exclude the outliers and compute the homography
[tform,inlierPtsB,inlierPtsA] = estimateGeometricTransform(matchedPointsA,matchedPointsB,'affine');

figure;
showMatchedFeatures(grayA,grayB,inlierPtsA,inlierPtsB,'montage');
title('Matched inlier points');

%% Prepare to warp the images
[w, h]     = deal(1400, 1000);  % Size of the mosaic
[x0, y0]   = deal(0, -200);   % Upper-left corner of the mosaic
xLim = [0.5, w+0.5] + x0;
yLim = [0.5, h+0.5] + y0;
outputView = imref2d([h,w], xLim, yLim);
mosaic = ones(h, w, 3, 'uint8')*255;

grayPics = [grayA,grayB];
colorPics = [colorA,colorB];
halphablender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
% for i = 1:length(grayPics)
%     mask = imwarp(ones(size(grayPics(i))), affine2d(eye(3)), 'OutputView', outputView);
%     transformedImage = imwarp(colorPics(i),  affine2d(eye(3)), 'OutputView', outputView);
%     
%     mask = mask >= 1;
%     mosaic = step(halphablender, mosaic, transformedImage, mask);
% end
mask = imwarp(ones(size(grayA(:,:,1))), affine2d(eye(3)), 'OutputView', outputView);
transformedImage = imwarp(colorA,  affine2d(eye(3)), 'OutputView', outputView);

mask = mask >= 1;
mosaic = step(halphablender, mosaic, transformedImage, mask);
mask = imwarp(ones(size(grayB(:,:,1))), affine2d(eye(3)), 'OutputView', outputView);
transformedImage = imwarp(colorB,  affine2d(eye(3)), 'OutputView', outputView);

mask = mask >= 1;
mosaic = step(halphablender, mosaic, transformedImage, mask);

figure;
J = imresize(mosaic, 0.5);
imshow(J);
title('Mosaic');


