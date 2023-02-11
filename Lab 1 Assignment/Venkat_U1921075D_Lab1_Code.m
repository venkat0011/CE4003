%% Section 2.1
clear; close all;
Pc = imread('mrttrainbland.jpg'); 
whos Pc
P = rgb2gray(Pc); 
whos P
figure;
imshow(P);
min(P(:)), max(P(:)) 

P_double = cast(P,"double");
P2 = ( 255 .* ( P_double - min(P_double(:)) ) ) ./ ...
                ( max(P_double(:)) - min(P_double(:))) ;
P2 = cast(P2,"uint8");
figure;
imshow(P2);
subplot(1,2,1);imshow(P);axis off; title("Original Image");
subplot(1,2,2);imshow(P2);axis off;title("Constrast Stretched Image");

min(P2(:))
max(P2(:))

%% section 2.2 Histogram equalisation
figure;
subplot(1,2,1);imhist(P,10);axis off; title("Intensity histogram with 10 bin");
subplot(1,2,2);imhist(P,256);axis off;title("Intensity histogram with 256 bins");
% so with 256 bins each bin will represent the pixel and we can get a better representation of the pixel distributions 
P3 = histeq(P,256); % the 256 refer to the number of bins you want to equalise so, if we plot with 256 it should be smoother that with 10 bins
figure;
subplot(1,3,1);imshow(P);axis off; title("Original Image");
subplot(1,3,2);imshow(P2);axis off;title("Stretched Image");
subplot(1,3,3);imshow(P3);axis off;title("Equalised Image");
figure;
subplot(1,2,1);imhist(P3,10);axis off; title(" Equalised histogram with 10 bin");
subplot(1,2,2);imhist(P3,256);axis off;title("Equalised histogram with 256 bins");
% what is the similarity -> both have lesser fluctations from before
% but the one with 10bins has lesser fluctuations that 256 WHY IS THAT
P3 = histeq(P3,256);
figure;
imshow(P3);
figure;
imhist(P3,10);
figure;
imhist(P3,256);
% as per the lecture notes re running histeq again will not change the
% result

% so we have 2 things to look out for, 1 is why the 10 and 256 differ and
% the other is why re applying will not work

% okay for the 256 bins it is normalised, and we can see that from the
% cumulative distibtion. It is almost linear implying that each bin has
% equal prob and we can also see that from the spacing, those bins that
% have a very high value are more spaced apart while those that have low
% values are very close together and this is why the flutates that we see
% in 256 bins is like averaged when reflecting in 10 bins


% histogram equalisation is idempotent so reapplying the same algo again
% will not change. Why? For histogram equalisation we are moving the bins
% with relative to all the previous bins to make the cdf as linear as
% possible. So the original image the histogram will be random but after
% equalisation the bins will be arrangesd in a way to make the cdf as
% linear as possible and since we will not move the amount of pixels in
% each bin reapplying the algo again will not change since the bins are
% already ion a order to linearise the cdf


%% section 2.3
%creating a 5 by 5 gaussian filter
% rmb that the 0,0 coordinates will refer to the centre of the filter 
% -1,0 will refer to the pixel on the left
% so if x and y are dimensions of 5 and 0 is the centre the values need to
% be from -2 to 2

sd = 1 ;
var = sd ^2;
convolutional_filter_1 = zeros(5,5);
for x = -2:2
    for y = -2:2
        convolutional_filter_1(x+3,y+3) = ( 1/ (2 * pi * var) ) * ...
                                    exp( -1*(x^2 + y^2)/(2*var)) ;
    end
end
convolutional_filter_1 = convolutional_filter_1 ./ sum(convolutional_filter_1(:));

sd = 2 ;
var = sd ^2;
convolutional_filter_2 = zeros(5,5);
for x = -2:2
    for y = -2:2
        convolutional_filter_2(x+3,y+3) = ( 1/ (2 * pi * var) ) * ...
                                    exp( -1*(x^2 + y^2)/(2*var)) ;
    end
end
convolutional_filter_2 = convolutional_filter_2 ./ sum(convolutional_filter_2(:));

figure;
subplot(1,2,1);mesh(convolutional_filter_1);axis off; title("Guassian kernel sd =1");
subplot(1,2,2);mesh(convolutional_filter_2);axis off;title("Guassian kernel sd =2");

image = imread('ntugn.jpg'); 
whos image

figure;
subplot(1,3,1);imshow(image);axis off; title("original image");
subplot(1,3,2);imshow(conv2(image,convolutional_filter_1),[]);;axis off;title("Guassian kernel sd =1");
subplot(1,3,3);imshow(conv2(image,convolutional_filter_2),[]);;axis off; title("Guassian kernel sd =2");

 % better to do subplot to see

% discuss about guassian filter -> what is their puprose what happens when
% the sigma inreases what do we see it here -> when is it more sutiable
% what we see here is that when the sigma increases the images get blurred
% more -> the borders of the image is heightened
% the sigma is the variation from the mean -> so a lower sigma would mean
% that the weights will be focused in the centre and less around 
% increasing the sigma will enable us to see the trends clearer but
% decrease the details and the effect of the noise
% Increasing the sigma would mean that the pixels further away from the
% centre will have a higher weight than before so the edge of the image is
% darkened and the sharpness of the iamge is lost.
% Gaussian filter assumes tyhat neighbouring pixels aare similar but in the
% case of edges its not true causing the edges to be blurred

image = imread('ntusp.jpg'); 
figure;
subplot(1,3,1);imshow(image);axis off; title("original image");
subplot(1,3,2);imshow(conv2(image,convolutional_filter_1),[]);axis off;title("Guassian kernel sd =1");
subplot(1,3,3);imshow(conv2(image,convolutional_filter_2),[]);axis off; title("Guassian kernel sd =2");

% it seems that the sigma 2 is better to remove the speckle noise but it is
% still very high bluriness (low sharpness)

%% section 2.4
image = imread('ntusp.jpg');
figure;
subplot(1,3,1);imshow(image);axis off; title("original image");
subplot(1,3,2);imshow(medfilt2(image,[3 3]),[]);axis off;title("3 by 3 median filter");
subplot(1,3,3);imshow(medfilt2(image,[5 5]),[]);axis off; title("5 by 5 median filter");


image = imread('ntugn.jpg');
figure;
subplot(1,3,1);imshow(image);axis off; title("original image");
subplot(1,3,2);imshow(medfilt2(image,[3 3]),[]);axis off;title("3 by 3 median filter");
subplot(1,3,3);imshow(medfilt2(image,[5 5]),[]);axis off; title("5 by 5 median filter");

% 3 by 3 is good but when it goes to 5 by 5 it blurs the lines, the window
% panes are dissappered
% so median filtering is a non linear methiod which is very effective in
% removing noise while perserving the edges ( gaussian doesnt preserve the
% edges)
% able to remove the outlier but for additive white guassian noise the
% noise is very similar to the original image so it is difficult to remove
% the noise 

%simialr to guassian it keeps the trend ( remove the high freq detials) 
% a larger kernel size will mean that it will encompose a larger set of
% pixels
%therefore the median value will deviate from the pixel value 


%% section 2.5
image = imread("pckint.jpg") ;
F = fft2(image);
S = abs(F);
figure;
subplot(1,3,1);imshow(image);axis off; title("original image");
subplot(1,3,2);imagesc((S.^0.1));axis off;title("Power specturm of image");
subplot(1,3,3);imagesc(fftshift(S.^0.1));axis off;title("Power specturm with FFTshift");



% the 2 freq are (241,9) and (249,17)
x1 = 241;x2 = 17;y1 = 9;y2 = 249;
% now we need to check if those 2 areas are indeed the high freq
figure;
imagesc(S(x1-2:x1+2,y1-2:y1+2).^0.1);
figure;
imagesc(S(x2-2:x2+2,y2-2:y2+2).^0.1);


%part d
F(x1-2:x1+2,y1-2:y1+2) = 0;
F(x2-2:x2+2,y2-2:y2+2) = 0;
figure;
subplot(1,2,1);imagesc((S.^0.1));axis off; title("original image power specturm");
subplot(1,2,2);imagesc((abs(F).^0.1));axis off;title("new power spectrum");
figure;
subplot(1,2,1);imshow(image);axis off;title("Original Image");
subplot(1,2,2);imshow(ifft2(F),[]);axis off;title("new Image");


% even though now we can see the characters and stuff mroe sharpely, there
% is still background noise that needs to be removed 
% there also needs to be some form of padding, 3 bits padding

F1 = fft2(image);
F1(x2:x1,y2) = 0;
F1(x2:x1,y1) = 0;
F1(x1,y1:y2) = 0;
F1(x2,y1:y2) = 0;


% needs abit more padding at the corners
F1(x1-3:x1+3,y1-3:y1+3) = 0;
F1(x2-3:x2+3,y2-3:y2+3) = 0;
figure;
figure;
subplot(1,2,1);imagesc((S.^0.1));axis off; title("original image power specturm");
subplot(1,2,2);imagesc((abs(F1).^0.1));axis off;title("new power spectrum");
figure;
subplot(1,3,1);imshow(image);axis off;title("Original Image");
subplot(1,3,2);imshow(ifft2(F),[]);axis off;title("new Image");
subplot(1,3,3);imshow(ifft2(F1),[]);axis off;title("Futher enhanced new Image");

%% section 2.5 part e
image =  imread('primatecaged.jpg');
whos image
% convert to grayscale
image = rgb2gray(image);
F = fft2(image);
temp = abs(F) .^ 0.1;
% so the cut off freq will be 3.2, anything greater than that has to be
% removed
temp = temp < 3.3 ;
temp1 = temp.* F ;

subplot(2,2,1);imshow(image);axis off;title("Original Image");
subplot(2,2,2);imagesc(fftshift(abs(F).^0.1));axis off;title("Specturm of original image wioth fftshift");
subplot(2,2,3);imshow(ifft2(temp1),[]);axis off;title("Futher enhanced new Image");
subplot(2,2,4);imagesc(fftshift(abs(temp1).^0.1));axis off;title("specturm of new image with fftshift");
%%
figure;
subplot(1,3,1);imshow(image);axis off;title("Original Image");
subplot(1,3,2);imshow(ifft2(F1),[]);axis off;title("Enhanced Image using previous method");
subplot(1,3,3);imshow(ifft2(temp1),[]);axis off;title("Enhanced Image using low pass filter");
%% section 2.5 but another method which doesnt blur the monkey
image =  imread('primatecaged.jpg');
whos image
% convert to grayscale
image = rgb2gray(image);
F = fft2(image);
F1 = F;
x1 = 251;x2 = 5;y1 = 11;y2 = 247;
x3 = 248;x4 = 11;y3 =22;y4 = 236;
F1(x1-2:x1+2,y1-2:y1+2) = 0;
F1(x2-2:x2+2,y2-2:y2+2) = 0;
F1(x3-2:x3+2,y3-2:y3+2) = 0;
F1(x4-2:x4+2,y4-2:y4+2) = 0;

subplot(2,2,1);imshow(image);axis off;title("Original Image");
subplot(2,2,2);imagesc(fftshift(abs(F).^0.1));axis off;title("Specturm of original image wioth fftshift");
subplot(2,2,3);imshow(ifft2(F1),[]);axis off;title("Futher enhanced new Image");
subplot(2,2,4);imagesc(fftshift(abs(F1).^0.1));axis off;title("specturm of new image with fftshift");
%% section 2.6
image =imread('book.jpg');
imshow(image);
[X, Y] = ginput(4);
x = [ 1 210 210 1];
y=  [ 1 1 297 297];

% A is the actual corner data while v is the potential corner data
%%
% getting the v matrix 
for i = 1: 8
    if(mod(i,2)==1)
        v(i) = x( (i+1) /2 );
    else
        v(i) = y(i/2);
    end
end
v = v' ;
% now get the A matrix
count = 1;
for i = 1:8
    if(mod(i,2)==0)
        A(i,:) = [ 0 0 0 X(count) Y(count) 1 -y(count)*X(count) -y(count)*Y(count)] ; 
        
        count = count+1;
    else
        A(i,:) = [ X(count) Y(count) 1 0 0 0 -x(count)*X(count) -x(count)*Y(count)] ; 
    end
end

u = A\v;
U = reshape([u;1], 3, 3)';
w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
T = maketform('projective', U');
P2 = imtransform(image, T, 'XData', [0 210], 'YData', [0 297]); 
figure;
subplot(1,2,1);imshow(image);axis off;title("Original Image");
subplot(1,2,2);imagesc(P2);axis off;title("Transformed Image");

% now we need to figure out why the thing is abit blurry

%% cropping the image using ginput
imshow(P2);
[Px,Py] = ginput(2) ; % select the top left and the btm right
R = [Px(1) Py(1)  Px(2)-Px(1) Py(2)-Py(1)];
I1 = imcrop(P2,R) ; 
figure
imshow(I1)
%% cropping the image usign image segmentation 

numColors = 8;
L = imsegkmeans(P2,numColors);
B = labeloverlay(P2,L);


lab_he = rgb2lab(P2);
ab = lab_he(:,:,2:3);
ab = im2single(ab);
pixel_labels = imsegkmeans(ab,numColors);
B2 = labeloverlay(P2,pixel_labels);


mask3 = pixel_labels == 2;
cluster3 = P2.*uint8(mask3);

% but there is some other stuff that needs to be remoevd
L = lab_he(:,:,1);
L_red = L.*double(mask3);
L_red = rescale(L_red);
idx_light_red = imbinarize(nonzeros(L_red));
red_idx = find(mask3);
red_mask = zeros(298,211);
red_mask(red_idx(idx_light_red)) = 1;

red_image = P2.*uint8(red_mask);

subplot(2,2,1);imshow(B);axis off;title("Clustering in RGB space");
subplot(2,2,2);imshow(B2);axis off;title("Clustering in L*a*b* space");
subplot(2,2,3);imshow(cluster3);axis off;title("Red Colour Mask");
subplot(2,2,4);imshow(red_image);axis off;title("Final Mask");