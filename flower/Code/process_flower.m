% Load Image
image= imread('flower.jpg');
image = imresize(image, [512 512]);

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image_hsv = rgb2hsv(image);
image_ycbcr = rgb2ycbcr(image);

% Thresholding in HSV (example: isolating pink regions)
hue = image_hsv(:,:,1);
sat = image_hsv(:,:,2);
val = image_hsv(:,:,3);

% Updated bg Detection for mid-tone to bright green
bg_mask = (hue > 0.25 & hue < 0.45) & (sat > 0.4 & val > 0.4); % Adjusted thresholds

% Object Detection for pink and yellow objects only
object_mask = ((hue > 0.9 | hue < 0.1) & sat > 0.4 & val > 0.6) | ...
               (hue > 0.1 & hue < 0.2 & sat > 0.4 & val > 0.4); % Adjusted thresholds for pink and yellow

pinkyellow_mask = (hue > 0.80 | hue < 0.2) & sat > 0.4 & val > 0.65; % Adjust thresholds as needed
% Morphological Processing to refine masks
se = strel('disk', 5); % Structuring element for morphological operations
bg_mask = imclose(bg_mask, se); % Close gaps in bg regions
bg_mask = imfill(bg_mask, 'holes'); % Fill holes in bg regions

object_mask = imclose(object_mask, se); % Close gaps in object regions
object_mask = imfill(object_mask, 'holes'); % Fill holes in object regions

% Overlay masks on original image for visualization
bg_overlay = labeloverlay(image, bg_mask, 'Colormap', [1 0 0], 'Transparency', 0.6); % Green for bgs
object_overlay = labeloverlay(image, object_mask, 'Colormap', [0 0 1], 'Transparency', 0.6); % Pink for objects

% Convert to grayscale
gray_image = rgb2gray(image);

% Apply Sobel edge detection
edges = edge(gray_image, 'sobel');

% Enhance edges using morphological dilation
edges_dilated = imdilate(edges, strel('line', 3, 90));

% Reshape image for K-means clustering
pixels = double(reshape(image, [], 3));
num_clusters = 3; % Number of clusters

[idx, cluster_centers] = kmeans(pixels, num_clusters);

% Reshape back to image dimensions
segmented_image = reshape(idx, size(image,1), size(image,2));

% Connected component analysis on pink mask (from color segmentation)
connected_components = bwconncomp(pinkyellow_mask);

% Extract region properties (centroid, area, bounding box)
stats = regionprops(connected_components, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image = labeloverlay(image, pinkyellow_mask);

% Display results
figure;

subplot(3,4,1);
imshow(image);
title('Original Image');

subplot(3,4,2);
imshow(pinkyellow_mask);
title('Pink, Yellow, and Red Regions Segmentation');

subplot(3,4,3);
imshow(edges_dilated);
title('Enhanced Edges');

subplot(3,4,4);
imagesc(segmented_image);
title('K-means Segmentation');

subplot(3,4,5);
imshow(image); hold on;
min_area = 50000;
for i = 1:length(stats)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    if bbox_area >= min_area
        rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end
title('Object Detection with Bounding Boxes');
hold off;

subplot(3,4,6);
imshow(overlay_image);
title('Segmentation Overlay (Flower Only)');

subplot(3,4,7);
imshow(edges);
title('Sobel Edges');

subplot(3,4,8);
imshow(hsv2rgb(image_hsv));
title('HSV Image');

subplot(3,4,9);
imshow(ycbcr2rgb(image_ycbcr));
title('YCbCr Image');

negative_image2 = 255 - image;
colored_image = image;

for c = 1:3
    colored_image(:,:,c) = uint8(pinkyellow_mask) .* negative_image2(:,:,c) + uint8(~pinkyellow_mask) .* image(:,:,c);
end

subplot(3,4,10);
imshow(colored_image);
title('Segmented Object with Changed Color');

% New figure just the object detection
figure;
imshow(image); hold on;

for i = 1:length(stats)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    if bbox_area >= min_area
        rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end

title('Object Detection with Bounding Boxes');
hold off;

segmented_image_rgb = label2rgb(segmented_image, 'jet', 'k', 'shuffle');

% Save the new figure
saveas(gcf, 'flower_object_detection_with_bounding_boxes.jpg');

imwrite(image, 'flower_original_image.jpg');
imwrite(pinkyellow_mask, 'flower_pink_yellow_segmentation.jpg');
imwrite(edges_dilated, 'flower_enhanced_edges.jpg');
imwrite(segmented_image_rgb, 'flower_kmeans_segmentation.jpg'); 
imwrite(overlay_image, 'flower_segmentation_overlay.jpg');
imwrite(edges, 'flower_canny_edges.jpg');
imwrite(hsv2rgb(image_hsv), 'flower_hsv_image.jpg');
imwrite(ycbcr2rgb(image_ycbcr), 'flower_ycbcr_image.jpg');
imwrite(colored_image, 'flower_segmented_object_colored.jpg');