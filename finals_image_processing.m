% Load Image
image = imread('car.jpg');

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image_hsv = rgb2hsv(image);
image_ycbcr = rgb2ycbcr(image);

% Thresholding in HSV (example: isolating red regions)
hue = image_hsv(:,:,1);
sat = image_hsv(:,:,2);
val = image_hsv(:,:,3);

% Shadow Detection based on lower brightness and saturation thresholds
shadow_mask = (sat < 0.7) & (val < 0.3); % Adjust thresholds as necessary

% Object Detection based on higher brightness and saturation thresholds
object_mask = (sat > 0.7) & (val > 0.3); % Adjust thresholds as necessary

red_mask = (hue > 0.10 | hue < 0.90) & sat > 0.2 & val > 0.1; % Adjust thresholds as needed

% Morphological Processing to refine masks
se = strel('disk', 5); % Structuring element for morphological operations
shadow_mask = imclose(shadow_mask, se); % Close gaps in shadow regions
shadow_mask = imfill(shadow_mask, 'holes'); % Fill holes in shadow regions

object_mask = imclose(object_mask, se); % Close gaps in object regions
object_mask = imfill(object_mask, 'holes'); % Fill holes in object regions

% Overlay masks on original image for visualization
shadow_overlay = labeloverlay(image, shadow_mask, 'Colormap', [0 0 1], 'Transparency', 0.6); % Blue for shadows
object_overlay = labeloverlay(image, object_mask, 'Colormap', [1 0 0], 'Transparency', 0.6); % Red for objects

% Convert to grayscale
gray_image = rgb2gray(image);

% Apply Canny edge detection
edges = edge(gray_image, 'sobel');

% Enhance edges using morphological dilation
edges_dilated = imdilate(edges, strel('line', 3, 90));

% Reshape image for K-means clustering
pixels = double(reshape(image, [], 3));
num_clusters = 3; % Number of clusters

[idx, cluster_centers] = kmeans(pixels, num_clusters);

% Reshape back to image dimensions
segmented_image = reshape(idx, size(image,1), size(image,2));

% Connected component analysis on red mask (from color segmentation)
connected_components = bwconncomp(red_mask);

% Extract region properties (centroid, area, bounding box)
stats = regionprops(connected_components, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image = labeloverlay(image, segmented_image);

% Create a figure with multiple subplots
figure;

% Subplot 1: Original Image
subplot(3,4,1);
imshow(image);
title('Original Image');

% Subplot 2: Red Regions Segmentation
subplot(3,4,2);
imshow(red_mask);
title('Red Regions Segmentation');

% Subplot 3: Enhanced Edges
subplot(3,4,3);
imshow(edges_dilated);
title('Enhanced Edges');

% Subplot 4: K-means Segmentation
subplot(3,4,4);
imagesc(segmented_image);
title('K-means Segmentation');

% Subplot 5: Object Detection with Bounding Boxes
subplot(3,4,5);
imshow(image); hold on;

% Set minimum area threshold for bounding boxes
min_area = 50000; % Adjust this value based on your requirements

for i = 1:length(stats)
    % Calculate the area of the bounding box (width * height)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    
    % Only plot the bounding box if the area is greater than the minimum area
    if bbox_area >= min_area
        rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end

title('Object Detection with Bounding Boxes');
hold off;

% Subplot 6: Segmentation Overlay
subplot(3,4,6);
imshow(overlay_image);
title('Segmentation Overlay');

% Subplot 7: Canny Edges
subplot(3,4,7);
imshow(edges);
title('Canny Edges');

% Subplot 8: HSV Image
subplot(3,4,8);
imshow(hsv2rgb(image_hsv));
title('HSV Image');

% Subplot 9: YCbCr Image
subplot(3,4,9);
imshow(ycbcr2rgb(image_ycbcr));
title('YCbCr Image');

% Subplot 5.1: Combined Overlay (Shadows and Objects)
combined_overlay = labeloverlay(image, shadow_mask | object_mask, ...
                                'Colormap', [1 0 0; 0 0 1], 'Transparency', 0.6);
subplot(3,4,10);
imshow(combined_overlay);
title('Combined Overlay (Shadows and Objects)');

% Compute the negative of the image
negative_image = 255 - image;

% Create a copy of the original image
colored_image = image;

% Apply the negative color only to the segmented object
for c = 1:3 % Loop through RGB channels
    colored_image(:,:,c) = uint8(red_mask) .* negative_image(:,:,c) + uint8(~red_mask) .* image(:,:,c);
end

% Display the new image
subplot(3,4,11);
imshow(colored_image);
title('Segmented Object with Changed Color');
