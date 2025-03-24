% Load Image
image1 = imread('car.jpg');

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image1_hsv = rgb2hsv(image1);
image1_ycbcr = rgb2ycbcr(image1);

% Thresholding in HSV (example: isolating red regions)
hue1 = image1_hsv(:,:,1);
sat1 = image1_hsv(:,:,2);
val1 = image1_hsv(:,:,3);

% Shadow Detection based on lower brightness and saturation thresholds
shadow_mask = (sat1 < 0.7) & (val1 < 0.3); % Adjust thresholds as necessary

% Object Detection based on higher brightness and saturation thresholds
object_mask1 = (sat1 > 0.7) & (val1 > 0.3); % Adjust thresholds as necessary

red_mask = (hue1 > 0.10 | hue1 < 0.90) & sat1 > 0.2 & val1 > 0.1; % Adjust thresholds as needed

% Morphological Processing to refine masks
se1 = strel('disk', 5); % Structuring element for morphological operations
shadow_mask = imclose(shadow_mask, se1); % Close gaps in shadow regions
shadow_mask = imfill(shadow_mask, 'holes'); % Fill holes in shadow regions

object_mask1 = imclose(object_mask1, se1); % Close gaps in object regions
object_mask1 = imfill(object_mask1, 'holes'); % Fill holes in object regions

% Overlay masks on original image for visualization
shadow_overlay = labeloverlay(image1, shadow_mask, 'Colormap', [0 0 1], 'Transparency', 0.6); % Blue for shadows
object_overlay1 = labeloverlay(image1, object_mask1, 'Colormap', [1 0 0], 'Transparency', 0.6); % Red for objects

% Convert to grayscale
gray_image1 = rgb2gray(image1);

% Apply Sobel edge detection
edges1 = edge(gray_image1, 'sobel');

% Enhance edges using morphological dilation
edges_dilated1 = imdilate(edges1, strel('line', 3, 90));

% Reshape image for K-means clustering
pixels1 = double(reshape(image1, [], 3));
num_clusters1 = 3; % Number of clusters

[idx, cluster_centers1] = kmeans(pixels1, num_clusters1);

% Reshape back to image dimensions
segmented_image1 = reshape(idx, size(image1,1), size(image1,2));

% Connected component analysis on red mask (from color segmentation)
connected_components1 = bwconncomp(red_mask);

% Extract region properties (centroid, area, bounding box)
stats1 = regionprops(connected_components1, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image1 = labeloverlay(image1, segmented_image1);

% Create a figure with multiple subplots
figure;

% Subplot 1: Original Image
subplot(3,4,1);
imshow(image1);
title('Original Image');

% Subplot 2: Red Regions Segmentation
subplot(3,4,2);
imshow(red_mask);
title('Red Regions Segmentation');

% Subplot 3: Enhanced Edges
subplot(3,4,3);
imshow(edges_dilated1);
title('Enhanced Edges');

% Subplot 4: K-means Segmentation
subplot(3,4,4);
imagesc(segmented_image1);
title('K-means Segmentation');

% Subplot 5: Object Detection with Bounding Boxes
subplot(3,4,5);
imshow(image1); hold on;

% Set minimum area threshold for bounding boxes
min_area = 50000; % Adjust this value based on your requirements

for i = 1:length(stats1)
    % Calculate the area of the bounding box (width * height)
    bbox_area = stats1(i).BoundingBox(3) * stats1(i).BoundingBox(4);
    
    % Only plot the bounding box if the area is greater than the minimum area
    if bbox_area >= min_area
        rectangle('Position', stats1(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end

title('Object Detection with Bounding Boxes');
hold off;

% Subplot 6: Segmentation Overlay
subplot(3,4,6);
imshow(overlay_image1);
title('Segmentation Overlay');

% Subplot 7: Sobel Edges
subplot(3,4,7);
imshow(edges1);
title('Sobel Edges');

% Subplot 8: HSV Image
subplot(3,4,8);
imshow(hsv2rgb(image1_hsv));
title('HSV Image');

% Subplot 9: YCbCr Image
subplot(3,4,9);
imshow(ycbcr2rgb(image1_ycbcr));
title('YCbCr Image');

% Subplot 5.1: Combined Overlay (Shadows and Objects)
combined_overlay1 = labeloverlay(image1, shadow_mask | object_mask1, ...
                                'Colormap', [1 0 0; 0 0 1], 'Transparency', 0.6);
subplot(3,4,10);
imshow(combined_overlay1);
title('Combined Overlay (Shadows and Objects)');

% Compute the negative of the image
negative_image1 = 255 - image1;

% Create a copy of the original image
colored_image1 = image1;

% Apply the negative color only to the segmented object
for c = 1:3 % Loop through RGB channels
    colored_image1(:,:,c) = uint8(red_mask) .* negative_image1(:,:,c) + uint8(~red_mask) .* image1(:,:,c);
end

% Display the new image
subplot(3,4,11);
imshow(colored_image1);
title('Segmented Object with Changed Color');

%############################################################################################################################################################
% Load Image
image2 = imread('flower.jpg');

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image2_hsv = rgb2hsv(image2);
image2_ycbcr = rgb2ycbcr(image2);

% Thresholding in HSV (example: isolating pink regions)
hue2 = image2_hsv(:,:,1);
sat2 = image2_hsv(:,:,2);
val2 = image2_hsv(:,:,3);

% Updated bg Detection for mid-tone to bright green
bg_mask = (hue2 > 0.25 & hue2 < 0.45) & (sat2 > 0.4 & val2 > 0.4); % Adjusted thresholds

% Object Detection for pink and yellow objects only
object_mask2 = ((hue2 > 0.9 | hue2 < 0.1) & sat2 > 0.4 & val2 > 0.6) | ...
               (hue2 > 0.1 & hue2 < 0.2 & sat2 > 0.4 & val2 > 0.4); % Adjusted thresholds for pink and yellow

pink_mask = (hue2 > 0.80 | hue2 < 0.1) & sat2 > 0.4 & val2 > 0.65; % Adjust thresholds as needed

% Morphological Processing to refine masks
se2 = strel('disk', 5); % Structuring element for morphological operations
bg_mask = imclose(bg_mask, se2); % Close gaps in bg regions
bg_mask = imfill(bg_mask, 'holes'); % Fill holes in bg regions

object_mask2 = imclose(object_mask2, se2); % Close gaps in object regions
object_mask2 = imfill(object_mask2, 'holes'); % Fill holes in object regions

% Overlay masks on original image for visualization
bg_overlay = labeloverlay(image2, bg_mask, 'Colormap', [1 0 0], 'Transparency', 0.6); % Green for bgs
object_overlay2 = labeloverlay(image2, object_mask2, 'Colormap', [0 0 1], 'Transparency', 0.6); % Pink for objects

% Convert to grayscale
gray_image2 = rgb2gray(image2);

% Apply Sobel edge detection
edges2 = edge(gray_image2, 'sobel');

% Enhance edges using morphological dilation
edges_dilated2 = imdilate(edges2, strel('line', 3, 90));

% Reshape image for K-means clustering
pixels2 = double(reshape(image2, [], 3));
num_clusters2 = 3; % Number of clusters

[idx, cluster_centers2] = kmeans(pixels2, num_clusters2);

% Reshape back to image dimensions
segmented_image2 = reshape(idx, size(image2,1), size(image2,2));

% Connected component analysis on pink mask (from color segmentation)
connected_components2 = bwconncomp(pink_mask);

% Extract region properties (centroid, area, bounding box)
stats2 = regionprops(connected_components2, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image2 = labeloverlay(image2, segmented_image2);

% Display results
figure;

subplot(3,4,1);
imshow(image2);
title('Original Image');

subplot(3,4,2);
imshow(pink_mask);
title('Pink Regions Segmentation');

subplot(3,4,3);
imshow(edges_dilated2);
title('Enhanced Edges');

subplot(3,4,4);
imagesc(segmented_image2);
title('K-means Segmentation');

subplot(3,4,5);
imshow(image2); hold on;
min_area = 50000;
for i = 1:length(stats2)
    bbox_area = stats2(i).BoundingBox(3) * stats2(i).BoundingBox(4);
    if bbox_area >= min_area
        rectangle('Position', stats2(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end
title('Object Detection with Bounding Boxes');
hold off;

subplot(3,4,6);
imshow(overlay_image2);
title('Segmentation Overlay');

subplot(3,4,7);
imshow(edges2);
title('Sobel Edges');

subplot(3,4,8);
imshow(hsv2rgb(image2_hsv));
title('HSV Image');

subplot(3,4,9);
imshow(ycbcr2rgb(image2_ycbcr));
title('YCbCr Image');

combined_overlay2 = labeloverlay(image2, bg_mask | object_mask2, 'Colormap', [1 0.9 0; 0 1 0.5], 'Transparency', 0.6);
subplot(3,4,10);
imshow(combined_overlay2);
title('Combined Overlay (BG and Objects)');

negative_image2 = 255 - image2;
colored_image2 = image2;

for c = 1:3
    colored_image2(:,:,c) = uint8(pink_mask) .* negative_image2(:,:,c) + uint8(~pink_mask) .* image2(:,:,c);
end

subplot(3,4,11);
imshow(colored_image2);
title('Segmented Object with Changed Color');

%############################################################################################################################################################
% Load Image
image3 = imread('Cat.jpg');

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image3_hsv = rgb2hsv(image3);
image3_ycbcr = rgb2ycbcr(image3);

% Thresholding in HSV (example: isolating white regions)
hue3 = image3_hsv(:,:,1);
sat3 = image3_hsv(:,:,2);
val3 = image3_hsv(:,:,3);

% Updated bg Detection for mid-tone to bright green
shadow2_mask = (hue3 > 0.01 | hue3 < 0.1) & (sat3 < 0.25) & (val3 < 0.40); % Adjusted thresholds

% Object Detection for white and brown objects only
object_mask3 = (hue3 > 0.01 | hue3 < 0.15) & sat3 < 0.4 & val3 > 0.65;

white_mask = (hue3 > 0.01 | hue3 < 0.15) & sat3 < 0.4 & val3 > 0.65; % Adjust thresholds as needed

% Morphological Processing to refine masks
se3 = strel('disk', 5); % Structuring element for morphological operations
shadow2_mask = imclose(shadow2_mask, se3); % Close gaps in bg regions
shadow2_mask = imfill(shadow2_mask, 'holes'); % Fill holes in bg regions

object_mask3 = imclose(object_mask3, se3); % Close gaps in object regions
object_mask3 = imfill(object_mask3, 'holes'); % Fill holes in object regions

% Overlay masks on original image for visualization
shadow2_overlay = labeloverlay(image3, shadow2_mask, 'Colormap', [1 0 0], 'Transparency', 0.6); 
object_overlay3 = labeloverlay(image3, object_mask3, 'Colormap', [0 0 1], 'Transparency', 0.6); 

% Convert to grayscale
gray_image3 = rgb2gray(image3);

% Apply Prewitt edge detection
edges3 = edge(gray_image3, 'prewitt');

% Enhance edges using morphological dilation
edges_dilated3 = imdilate(edges3, strel('line', 3, 90));

% Reshape image for K-means clustering
pixels3 = double(reshape(image3, [], 3));
num_clusters3 = 3; % Number of clusters

[idx, cluster_centers3] = kmeans(pixels3, num_clusters3);

% Reshape back to image dimensions
segmented_image3 = reshape(idx, size(image3,1), size(image3,2));

% Connected component analysis on white mask (from color segmentation)
connected_components3 = bwconncomp(white_mask);

% Extract region properties (centroid, area, bounding box)
stats3 = regionprops(connected_components3, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image3 = labeloverlay(image3, segmented_image3);

% Display results
figure;

subplot(3,4,1);
imshow(image3);
title('Original Image');

subplot(3,4,2);
imshow(white_mask);
title('White Regions Segmentation');

subplot(3,4,3);
imshow(edges_dilated3);
title('Enhanced Edges');

subplot(3,4,4);
imagesc(segmented_image3);
title('K-means Segmentation');

subplot(3,4,5);
imshow(image3); hold on;
min_area = 50000;
for i = 1:length(stats3)
    bbox_area = stats3(i).BoundingBox(3) * stats3(i).BoundingBox(4);
    if bbox_area >= min_area
        rectangle('Position', stats3(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end
title('Object Detection with Bounding Boxes');
hold off;

subplot(3,4,6);
imshow(overlay_image3);
title('Segmentation Overlay');

subplot(3,4,7);
imshow(edges3);
title('Prewitt Edges');

subplot(3,4,8);
imshow(hsv2rgb(image3_hsv));
title('HSV Image');

subplot(3,4,9);
imshow(ycbcr2rgb(image3_ycbcr));
title('YCbCr Image');

combined_overlay3 = labeloverlay(image3, shadow2_mask | object_mask3, 'Colormap', [1 0.9 0; 0 1 0.5], 'Transparency', 0.6);
subplot(3,4,10);
imshow(combined_overlay3);
title('Combined Overlay (Shadow and Objects)');

negative_image3 = 255 - image3;
colored_image3 = image3;

for c = 1:3
    colored_image3(:,:,c) = uint8(white_mask) .* negative_image3(:,:,c) + uint8(~white_mask) .* image3(:,:,c);
end

subplot(3,4,11);
imshow(colored_image3);
title('Segmented Object with Changed Color');
