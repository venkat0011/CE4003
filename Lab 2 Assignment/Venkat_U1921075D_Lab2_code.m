%% section 3.1 part a ======================================
clear; close all;
image  = imread('macritchie.jpg'); 
whos image
image = rgb2gray(image); 
figure;
imshow(image);

% creating a 3x3 sobel mask 
% horizontal
sobel_horizontal = [-1 -2 -1 ; 0 0 0 ; 1 2 1] ;
sobel_vertical = [-1 0 1; -2 0 2; -1 0 1 ];

subplot(1,2,1);imshow(conv2(image,sobel_vertical),[]);axis off; title("Vertical Edges");
subplot(1,2,2);imshow(conv2(image,sobel_horizontal),[]);axis off;title("Horizontal Edges");
% the horizontal sobel was able to detect the horizontal line like the pine
% tree and the lines on the building
% the vertical sobel was able to detect the vertical lines like the outline
% of the people -> an interesting finding is that it can detect one side of
% the top of the building but not the other 
% and another interesting finding is that the pathway which is diagonal
% appears on both the vertical and horizontal convolutions but it appears
% to be stronger on the horizontal part. 
% if we look at a vertical line there is a horizontal gradient shift
% and in a horiztonal line there is a vertical gradient shift
% but in the case of diagonal lines, there is both a vertical shift in the
% gradient and a horizontal shift in the gradient that is why it is able to
% appear on both sobel convolutions

horizontal_edge = conv2(image,sobel_horizontal);
vertical_edge = conv2(image,sobel_vertical);
magnitude = (horizontal_edge .^2) + (vertical_edge.^2);
figure;
imshow(magnitude);
% by squaring the image what we are finding is hte absolute gradient
% magnitude, cos gradient can be positive or negative but we are only
% concerned with the magnitude of these gradients for edge detection 

% experimenting with different thresholds 
% the range is from 0 to 10 * 10^5
threshold_array = [0 10^1 10^2 10^3 10^4 10^5];
count = 1;
figure;
for i = threshold_array
    subplot(2,3,count);imshow(magnitude>i,[]);axis off;title("Threshold = "+i);
    count=count+1;
end

% first discuss the purpose of having a threshold
% by having the magnitude edge image it combines all the edges found in the
% image be it small or large gradient changes. Certain edges have a sharp
% graident change while some have a less steeper gradient chagne. BY having
% a threshold we are able to remove the edges that have a magnitude lesser
% than T. And it can be seen from our experiment here. As the T increases
% the number of lines shown decreases and it only picks up the edges that
% have a very stark gradient change But what we also see is that some of
% the small edges that give the details of the image are lost as the
% threshold increases.
% so in conclusion having a lower threshold will enable use to pick the
% details of the image but it will also select the edges that are
% irrelevant increasing the noise of the image
% having a larger threshold we will only select the strong lines and remove
% the noisy edges but we will risk losign the details of the image 
% so a optimal threshold would be one that picks out the detials of the
% image but ignoring the noisy edges 
%%
% question e
tl = 0.04; th=0.1; sigma=1.0;
% E = edge(image,'canny',[tl th],sigma);
figure;

for i = 1:1:5
    E = edge(image,'canny',[0.04 0.1],i);
    subplot(2,3,i);imshow(E);axis off;title("Sigma = "+i);
end
E = edge(image,'canny',[tl th],5.5);
subplot(2,3,6);imshow(E,[]);axis off;title("Sigma = "+5.5);

% when the sigma is increased what we see is that the number of lines are
% decreasing, the edges that give detail to the image such as the outline
% of the people and the horizontal lines in the structure are removed.

% so why is this ? What is the pupose of sigma in canny edge
% canny edge is broken into 3 parts, guassian edge filtering, non maximal
% supression and hystersis thresholding
% now guassian edge filtering is filtered twice by the x and y derivatives
% of guassian, by increasing the sigma, we will increase the spread of the
% guassian distribution and it will filter out =the details only keeping
% the trends. Therefore we see that the number of edges is reduced and we
% see that the location o fthese edgels are less accurate with the removal
% of details

% for noisy edgel removal having a high sigma will be suitable since we
% only want to keep the trend and remove all the noise. But if we want to
% locate these edgels accurately then we need to have a low sigma 
%%
figure;
count = 1;
for i= 0.04:0.01:0.09
    E = edge(image,'canny',[i 0.1],1);
    subplot(2,3,count);imshow(E);axis off;title("tl = "+i);
    count = count+1;
end

% what does tl represent the lower threshold while the th represent the
% higher threholding. Previosuly we mentioned that canny edge uses
% hysteresis thresholding. Anything above the higher threshold will be a 1 and anything below the lower threshold will be classified as 0
% any pixel that has a value between those will see if the neighbouring pixels that were perpendiculat to the edge gradient has been set to 1 or 0 and it will follow that 
% By increasing the the lower threshold the space
% between the higher and lower threshold decreases, this will reduce the
% tiny noisy edges and find the long edges as we can see from the changes
% in the image. THe line of the branch that is nearere to the tree is seen
% decreasing
%%
tl = 0.04; th=0.1; sigma=1.0;
E = edge(image,'canny',[tl th],sigma);
[H, xp] = radon(E); 

figure;
imagesc(0:179,xp,H);colormap(gray);xlabel("thetha");ylabel("p");
% identify the similarities between hough transform and random transform
% first we need to discuss what is hough transform, and how it is performed
% hough transform is usually used for edge linking : to determine which
% lines should connect these 2 edges that are detected
% so what we do for hough transform is to break the lines into the product
% of a vector and the line that is perpendicular to it ( from the origin)
% the intuition for this is cos is that every vector on the line must be perpendicular (orthogonal) to the straight line of length r that comes from the origin.
% so each lines can be rewritten as rough and theta,so for a given
% coordinate there can be multiple pair of values that can satisfy it.
% to deteermine the areas where the lines interesect in the hough space
% an accumulator is increasemented when the lines passes through that point
% in hough space. So the areas where the accumulator has high values would
% be the optimal p and theta to join the lines together

% FOr randon tranform it is a projection of the image intensity along a radial line oriented at a
% specific angle. SO what it does is that it will split each pixel into 4
% sub pixel then project each of these subpixel seperately. IF the
% subpixels project hits the center point of a bin the bin on the aces will
% get the full value of the subpizxel which is 1/4 of the main pixel, if
% the projection hits the border the sub pixel is split evenly between the
% bins. These bins that the matlab is talking about refer to the detector
% line https://www.youtube.com/watch?v=MA2y_2YySq0&ab_channel=ASTRAToolbox
% we are also rotating this detector line for all possible theta values
% https://digitalcommons.colby.edu/cgi/viewcontent.cgi?article=1649&context=honorstheses
% so iof the points are all collinear, then these points will have the
% projection falling into the same bin. Therefore the value of theta and t
% will have a higher intensity values( since there will be a higher sum of
% 1/4 pixels) when compared to the rest. This method is very similar to the
% voting method that hough transform takes ( accumulator)

% similar to hough transform the lines in radon transform are expressed
% using theta and t is the result of "point normal" parameterization of a
% line 

% random and hough have a similar methods, it can be viewed that randon is
% the we consider how a data point in the destination space is obtained from the
% data in the source space: the reading paradigm so it derives a point in
% the parametric space from image space

% for hough it is how a data point in the source space
% maps onto data points in the destination space: the writing paradigm
%https://web.archive.org/web/20160729172119/http://tnw.home.tudelft.nl/fileadmin/Faculteit/TNW/Over_de_faculteit/Afdelingen/Imaging_Science_and_Technology/Research/Research_Groups/Quantitative_Imaging/Publications/Technical_Reports/doc/mvanginkel_radonandhough_tr2004.pdf
% so a hough transform is a discretisation of the cointinous randon
% transform. But this is only possible in bimary images https://sci-hub.se/10.1364/ao.54.010586
%%
% find the value for theta and p for the max intensity
max_int= max(H(:));
[p_index,theta] = find(H==max_int);
radius = xp(p_index) ;
theta = theta -1 ;% cos the index starts from 1 but theta the range is from 0
disp(theta)
disp(radius)
%%
% we need to show hpw pol2cart is able to give us the same formula as the
% hough  transform

[A,B] = pol2cart(theta*pi/180, radius);
B = -B;
C = radius ^2 + 179*A+145*B;
% as mentioned previouslt for hough transform the equation is xcos +y sin
% =p so a here is cos and b is sin and c is p 
% for pol2cart we are converting the polar to a cartesian mapping so the x
% is value will be rcostheta and y will be sin, here the thetha needs to be
% in radian thats why we divide by 180 and times by pi
% now when we use pol2cart the result return is a mutliplication of p but
% if we compare the line equation a should just be cos theta so we have to
% multiply p to the line equation so c will represent p2. Another thing to
% take note is that hough transform is done with repsect to the origin so
% we will need to convert the origin of the image which is located at the
% top left of the iamge to the centre this can be done by translating
% positive 179 units in the x and 145 units in the y direction. So the new
% equation is A(x-179) + B(y-145) = p^2 , so simplyfiying the equation and
% bringing the constant over what we get is that C is p^2 +179A+14B
%%
xl = 0;
yl =( C - (A*xl) )/ B;
xr = length(image) - 1;
yr = ( C - (A*xr) )/ B;
figure;
imshow(image,[]);
line([xl xr], [yl yr]); 
%%
% the line does not match the running track exactly there are some
% deviations at the top and the middle
% so some areas where there could be error is the conversion from paramter
% space to image space, and since the line seems to deviate away from the
% path at larger distances the path might not exactly be one straight line
% it might be curved. To accomadate the later path of the path it seems the
% theta has to be lesser, but now the front part of the path way deviates.
% If we take a mid point of the previous y and the new y we will get a line
% that fits the later half but deviates from the first half. SO we can
% really assign equal weightage. So after many manual trials, it seems the
% approporiate weightage is to give the first line 80% and the second line
% 20% by doing so the line mainly follows the first line but it will also
% have some characterisitic of the second line and the line fits more
% accurately. So in the future events a possible way is to come out with a
% piece wise function so that non linear lines can be mapped so we can
% break the image into smaller segments/ smaller patches and do the thing
% and piece the lines together instead
%%
tl = 0.04; th=0.1; sigma=1.0;
E = edge(image,'canny',[tl th],sigma);
[H, xp] = radon(E); 
max_int= max(H(:));
[p_index,theta] = find(H==max_int);
radius = xp(p_index) ;
theta = theta  -2;% cos the index starts from 1 but theta the range is from 0
disp(theta)
disp(radius)
[A,B] = pol2cart(theta*pi/180, radius);
B = -B;
C = radius ^2 + 179*A+145*B;
x2 = 0;
y2 =( C - (A*x2) )/ B;
x3 = length(image) - 1;
y3 = ( C - (A*x3) )/ B;
y2 = y2* 0.2 + yl * 0.8;
y3 = y3*0.2 + yr*0.8;
figure;
imshow(image,[]);
line([x2 x3], [y2 y3]); 

%% section 3 3d 
% left_image = imread('corridorl.jpg');
% left_image = rgb2gray(left_image);
% right_image = imread('corridorr.jpg');
% right_image = rgb2gray(right_image);
% disp_map = imread("corridor_disp.jpg");
% figure;
% subplot(1,2,1);imshow(left_image,[]);axis off;title("left image");
% subplot(1,2,2);imshow(right_image,[]);axis off;title("right image");
% D = generate_disp_map(left_image,right_image,11,11);
% figure;
% subplot(1,2,1);imshow(D,[-15 15]);axis off;title("Calculated Disaprity Map");
% subplot(1,2,2);imshow(disp_map,[]);axis off;title("Actual Disparity Map");




left_image = imread('triclopsi2l.jpg');
left_image = rgb2gray(left_image);
right_image = imread('triclopsi2r.jpg');
right_image = rgb2gray(right_image);
disp_map = imread("triclopsid.jpg");
figure;
subplot(1,2,1);imshow(left_image,[]);axis off;title("left image");
subplot(1,2,2);imshow(right_image,[]);axis off;title("right image");
D = generate_disp_map(left_image,right_image,11,11);
figure;
subplot(1,2,1);imshow(D,[-15 15]);axis off;title("Calculated Disaprity Map");
subplot(1,2,2);imshow(disp_map,[]);axis off;title("Actual Disparity Map");


% left_image = imread('triclopsi2l.jpg');
% left_image = rgb2gray(left_image);
% right_image = imread('triclopsi2r.jpg');
% right_image = rgb2gray(right_image);
% D = generate_disp_map(left_image,right_image,11,11);
% figure
% imshow(D,[-15 15]);colormap('gray');





function disparity_map = generate_disp_map(left_image,right_image,template_height,...
                                            template_width)
        
         % in the left image extract the 11x11 neighbourhood
         [image_height,image_width] = size(left_image);
          % intialise the return array
         % for the 11 by 11 pixels not all the pixels can compute the
         % disparity, only those that have the 11 by 11 elements
         % surrounding it can find the disparity
         x_padding = floor(template_width/2);
         y_padding = floor(template_height/2);
         disparity_map = ones(image_height-template_height+1,image_width-template_width+1);
         for y_pixel = 1+y_padding:image_height-y_padding % we are doing y loop first cos we are going across different values of x to find the correspondance that has the lowest SSD
             for x_pixel = 1+x_padding:image_width-x_padding
                 % we extract the template of the pixel
                 left_template = left_image(y_pixel-y_padding:y_pixel+y_padding,x_pixel-x_padding:x_pixel+x_padding); % so the x and y pixel are the middle of the template
                 % now we want to traverse horizontally but not too much we
                 % want the disparity to be within 15
                 % for the first pixel the min value of x is 1+x_padding,
                 % but for other pixel the 1+x_padding is more than 15
                 % pixels away so we need to find the maximum between these
                 % 2, to prevent any error in the event that x_pixel-15
                 % leads to zero
                 min_x = max(1+x_padding,x_pixel-15);
                 max_x = min(image_width-x_padding,x_pixel+15);
                 % now we need to compute the disparity of these x pixels
                 % given a y value
                 min_ssd = inf;
                 min_ssd_index = x_pixel; % we need to monitor where the correspondance is for the right image
                 for x = min_x:max_x
                     % extract the same template in the right image but now
                     % the x will be the centre value and we will be
                     % calculating the ssd of this template in the right
                     % image to the left image within the possible values
                     % of x and calculate the one with the minimum ssd
                     right_template = right_image(y_pixel-y_padding:y_pixel+y_padding,x-x_padding:x+x_padding); 
                     temp = rot90(right_template,2);
                     first_term = conv2(right_template,temp,'same');
                     last_term = 2.*conv2(temp,left_template,'same');
                     ssd = first_term(y_padding+1,x_padding+1) - last_term(y_padding+1,x_padding+1); % the template will be a 11 by 11 matrix, we cannot use the whole range, so ssd will also be 11x11 and we are concerned with the ssd of the centre pixel
                     if(ssd<min_ssd)
                         min_ssd = ssd;
                         min_ssd_index = x;
                     end
                 end
                     disparity_map(y_pixel-y_padding,x_pixel-x_padding) = x_pixel - min_ssd_index; % calculating disparity 
                     % we minus the padding here so that those edge pixel
                     % that we missed out will have a value 
             end
         end
end

% we can see most of the data is the same except the wall, the wall at the
% end of the alley is just a white wall and it is a homogenous area and for
% homogenous areas it is very difficult to find the exact corresponding
% pixel since there can be many pixel with similar ssd so it might take the
% first pixel that has the lowest ssd and the actual correspondance might
% be further to the right. Another error could be the restriction in
% scanline, the correspondance might fall more than these scaline of 15 and
% it might not even fall along the scanline even.


% section d

%now the image looks identicular but we are concerned with the accuracy of
%the disparity. When we measured the disparity in corridor it was a smooth
%decrease in disparity but when if we refer here the pathway that is seen
%nearer to the camer has fluctuating disparity some are high some are low
% if we look at the images side by side it doesnt seem that the camera
% translated just right or left, it seems the angle that the camera was
% facing changed -> so we are unable to just find the correspondance by
% comparing the pixel along the scan line since there is a shift in
% vertical and horizontal and it seems the camera itself was shifted so
% maybe we need to find the rectification homogropy to project these image
% plane to parallel to one another then we can find the depth map using the
% same method

% another possible reason of error is the one where we are ussing SSD which
% is appearance based matching, not very accurate should use feature based
% mapping

                 
