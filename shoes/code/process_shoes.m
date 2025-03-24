% Load Image
image = imread('shoes.jpg');
image = imresize(image, [512 512]);

%-----------------------------------------------------------------------
% Convert to HSV and YCbCr color spaces
image_hsv = rgb2hsv(image);
image_ycbcr = rgb2ycbcr(image);

% Thresholding in HSV
hue = image_hsv(:,:,1);
sat = image_hsv(:,:,2);
val = image_hsv(:,:,3);
cutoff_row = round(size(image, 1) * 0.75);

% Brown color segmentation (tuned for shoe color)
red_mask = (hue > 0.02 & hue < 0.18) & sat > 0.2 & val > 0.0001; % Adjusted thresholds for brown shoes
red_mask(cutoff_row:end, :) = 0; % Set lower part to black

% Black color segmentation (for pants and socks)
pants_mask = val < 0.25; % Adjust threshold as needed - increased slightly
pants_mask(cutoff_row:end, :) = 0; % Set lower part to black

% Combine the masks for the Region Segmentation subplot
combined_mask = red_mask | pants_mask;


% Morphological Processing to refine the red mask (focus on shoe shape)
se = strel('disk', 10); % Larger disk for closing gaps
red_mask = imclose(red_mask, se); % Close gaps in shoe regions
red_mask = imfill(red_mask, 'holes'); % Fill holes in shoe regions
se2 = strel('disk', 5);
red_mask = imerode(red_mask, se2); % Erode to separate close objects

% Shadow Detection (adjusted thresholds for shoe image)
shadow_mask = (sat < 0.4) & (val < 0.4); % Adjusted thresholds

% Object Detection (adjusted thresholds for shoe image)
object_mask = (sat > 0.3) & (val > 0.4); % Adjusted thresholds

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

% Connected component analysis on the red mask
connected_components = bwconncomp(red_mask);

% Extract region properties (centroid, area, bounding box)
stats = regionprops(connected_components, 'Centroid', 'Area', 'BoundingBox');

% Overlay segmentation output on original image
overlay_image = labeloverlay(image, red_mask);

% Create a figure with multiple subplots
figure;

% Subplot 1: Original Image
subplot(3,4,1);
imshow(image);
title('Original Image');

% Subplot 2: Brown and Black Segmentation (Pants and Shoes)
subplot(3,4,2);
imshow(combined_mask);
title('Brown and Black Segmentation (Pants and Shoes)');

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

% Set minimum and maximum area thresholds for bounding boxes
min_area = 1000;  % Adjusted based on shoe size in the image
max_area = 50000; % Adjusted based on shoe size in the image

for i = 1:length(stats)
    % Calculate the area of the bounding box (width * height)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    
    % Only plot the bounding box if the area is within the defined range
    if bbox_area >= min_area && bbox_area <= max_area
        rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end

title('Object Detection with Bounding Boxes');
hold off;

% Subplot 6: Segmentation Overlay
subplot(3,4,6);
imshow(overlay_image);
title('Segmentation Overlay (Shoes Only)');

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

% Subplot for segmented object
min_area = 1000;  % Adjusted based on shoe size in the image
max_area = 50000; % Adjusted based on shoe size in the image

% Compute the negative of the image
negative_image = 255 - image;

% Copy of the original image
colored_image = image;

% Loop through detected objects
for i = 1:length(stats)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    
    % Apply negative transformation only if object area is within the range
    if bbox_area >= min_area && bbox_area <= max_area
        % Extract bounding box coordinates
        bbox = round(stats(i).BoundingBox);
        x_start = bbox(1);
        y_start = bbox(2);
        width = bbox(3);
        height = bbox(4);
        
        % Ensure indices are within valid image bounds
        x_end = min(x_start + width, size(image, 2));
        y_end = min(y_start + height, size(image, 1));
        
        % Apply the negative color only to the segmented object within the bounding box
        for c = 1:3 % Loop through RGB channels
            colored_image(y_start:y_end, x_start:x_end, c) = ...
                uint8(red_mask(y_start:y_end, x_start:x_end)) .* ...
                negative_image(y_start:y_end, x_start:x_end, c) + ...
                uint8(~red_mask(y_start:y_end, x_start:x_end)) .* ...
                image(y_start:y_end, x_start:x_end, c);
        end
    end
end

subplot(3,4,10);
imshow(colored_image);
title('Segmented Object with Changed Color (Shoes Only)');

% New figure just the object detection
figure;
imshow(image); hold on;

for i = 1:length(stats)
    bbox_area = stats(i).BoundingBox(3) * stats(i).BoundingBox(4);
    if bbox_area >= min_area && bbox_area <= max_area
        rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
end

title('Object Detection with Bounding Boxes');
hold off;

% Save the new figure
saveas(gcf, 'shoes_object_detection_with_bounding_boxes.jpg');

imwrite(image, 'shoes_original_image.jpg');
imwrite(combined_mask, 'shoes_brown_black_segmentation.jpg');
imwrite(edges_dilated, 'shoes_enhanced_edges.jpg');
imwrite(segmented_image_rgb, 'shoes_kmeans_segmentation.jpg'); 
imwrite(overlay_image, 'shoes_segmentation_overlay.jpg');
imwrite(edges, 'shoes_canny_edges.jpg');
imwrite(hsv2rgb(image_hsv), 'shoes_hsv_image.jpg');
imwrite(ycbcr2rgb(image_ycbcr), 'shoes_ycbcr_image.jpg');
imwrite(colored_image, 'shoes_segmented_object_colored.jpg');
