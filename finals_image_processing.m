% Load Image
image = imread('car.jpg');

% Convert to HSV and YCbCr color spaces
image_hsv = rgb2hsv(image);
image_ycbcr = rgb2ycbcr(image);

% Thresholding in HSV (example: isolating red regions)
hue = image_hsv(:,:,1);
sat = image_hsv(:,:,2);
val = image_hsv(:,:,3);

red_mask = (hue > 0.95 | hue < 0.05) & sat > 0.5 & val > 0.5; % Adjust thresholds as needed
%-----------------------------------------------------------------------
% Display segmented regions
figure; imshow(red_mask); title('Red Regions Segmentation');

% Convert to grayscale
gray_image = rgb2gray(image);

% Apply Canny edge detection
edges = edge(gray_image, 'canny');ah

% Enhance edges using morphological dilation
edges_dilated = imdilate(edges, strel('line', 3, 90));

% Display edges
figure; imshow(edges_dilated); title('Enhanced Edges');

%-----------------------------------------------------------------------
% Reshape image for K-means clustering
pixels = double(reshape(image, [], 3));
num_clusters = 3; % Number of clusters

[idx, cluster_centers] = kmeans(pixels, num_clusters);

% Reshape back to image dimensions
segmented_image = reshape(idx, size(image,1), size(image,2));

% Display segmented image
figure; imagesc(segmented_image); title('K-means Segmentation');

%-----------------------------------------------------------------------
% Connected component analysis on red mask (from color segmentation)
connected_components = bwconncomp(red_mask);

% Extract region properties (centroid, area, bounding box)
stats = regionprops(connected_components, 'Centroid', 'Area', 'BoundingBox');

% Overlay bounding boxes on original image
figure; imshow(image); hold on;
for i = 1:length(stats)
    rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    text(stats(i).Centroid(1), stats(i).Centroid(2), sprintf('%d', i), ...
        'Color', 'yellow', 'FontSize', 12);
end
title('Object Detection with Bounding Boxes');
%-----------------------------------------------------------------------
% Feature Extraction: Color Histogram and Texture (GLCM)
gray_image = rgb2gray(image);

% Color histogram (example: red channel)
red_channel = image(:,:,1);
color_histogram = imhist(red_channel);

% GLCM texture features
glcm = graycomatrix(gray_image);
texture_features = graycoprops(glcm);

% Train SVM or KNN classifier (requires labeled dataset)
% Example: Using pre-trained model for classification (pseudo-code)
scene_label = predict(trained_model, [color_histogram(:); texture_features.Contrast]);
disp(['Scene Classification: ', scene_label]);

%-----------------------------------------------------------------------
% Overlay segmentation output on original image
overlay_image = labeloverlay(image, segmented_image);

figure; imshow(overlay_image); title('Segmentation Overlay');
%-----------------------------------------------------------------------
layers = [
    imageInputLayer([size(image,1) size(image,2) 3])
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10) % Example: 10 classes
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10);

trained_net = trainNetwork(training_images, training_labels, layers, options);

predicted_label = classify(trained_net, image);
disp(['Predicted Scene: ', char(predicted_label)]);
