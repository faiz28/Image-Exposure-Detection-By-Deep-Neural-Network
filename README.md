# Image Exposure Detection By Deep Neural Network

Collect our dataset from this link : [Dataset](https://drive.google.com/drive/folders/1PpyQIIqjv7d9lVfZhUlLHDguKxImX1xQ?usp=sharing)


*[Faiz Ahmed](https://github.com/faiz28)*<sup>1</sup>, 
*[Tasrin Sultana](https://github.com/TasrinSultana)*<sup>1</sup>


## Dataset
To do our work, we need a large number of images rendered with realistic over- and under-exposure errors and corresponding properly exposed ground truth images. However, such data sets are currently not publicly available. We have found that the MIT-Adobe FiveK dataset is the closest publicly available data set which can full fill the requirements of our work. In this data set, there are 5,000 photographs captured with SLR cameras by a set of different photographers. All the photographs are in RAW format. That is, all the information recorded by the camera sensor is preserved. It is made sure that these photographs cover a broad range of scenes, subjects, and lighting conditions. Five photography students in an art school were hired to adjust the tone of the photos. They retouched each of the photo by hand. Each of them retouched all the 5,000 photos using a software (named Adobe Lightroom) dedicated to photo adjustment on which they were extensively trained. They rendered each raw-RGB image with different digital EVs (Exposure Values) to mimic real exposure errors. Specifically, they used the relative {\bf{EVs −1.5, −1, +0, +1, and +1.5 to render images with underexposure errors, a zero gain of the original EV, and overexposure errors}}, respectively. For our experiment we take -1.5 is underexposed images, +0 is normal images and +1.5 is a overexposed images. 
![dataset](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/dataset.png)

We apply different data processing technique technique on our data set, then apply transfer learning and our own CNN model on our dataset.

### Different data processing technique
1. Single channel image (i.e, Red, Green,Blue)
2. Negative images
3. Histograms of Gray Scale Images
4. Images with Increased and Decreased Exposure

# Single channel image
![rgb](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/red_green_blue_channel_of_photo%20(1).png)


# Negative Images
![negative](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/negativeImage%20(1).png)



# Histograms of Gray Scale Images
![Histogram](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/histogram%20image.png)




# Images with Increased and Decreased Exposure
![overexposed](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/more-overexposed.png)


![underexposed](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/more-underexposed.png)

### We have developed five models:
1. Fully fine-tuned VGG-16 Model
2. Partially fine-tuned VGG-16 Model
3. Fully Connected Neural Network (FCNN)
4. Convolutional Neural Network-1 (CNN-1)
5. Convolutional Neural Network-2 (CNN-2)

### Result 
![accuracy](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/Screenshot%20from%202022-04-21%2002-08-49.png)

### Reason of low accurcy
![reason](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/confusion-label%20(1).png)
The underlying reasons behind the mis-classifications can be better understood. In this figure, the confusion matrix is shown with a few randomly selected sample images. First, it can be seen that most of the misclassified  overexposed images taken in low light and underexposed images taken in high light. Pictures taken in low light environment contains less exposure compared to normal image(which is taken in proper light environment) that is why sometimes model misclassifies the image by predicting as underexposed image. In our data set, there are many images which are very difficult to correctly classify even by human. Many of our over-exposed labelled image have low exposure which are more likely normal image. And many of our normal labelled image have not correct exposure that is why some of them looked as over-exposed image and also some of them looked as under-exposed image. Just like over-exposed and normal image, many of our under-exposed image looks like properly exposed image.

### Conclusion
1. We have developed deep-learning based classifiers to decide whether an image is under-exposed, over-exposed or normal.
2. We have investigated the performance of all models by different types of pre-processed image.
3. We have achieved the best performance (90%) on images with increased and decreased exposure using the FCNN model.


For more details please  check our written details here ![project paper](https://github.com/faiz28/Image-Exposure-Detection-By-Deep-Neural-Network/blob/main/images/Detecting_Inappropriate_Exposure_in_Image_by_Deep_Neural_Networks%20(6)-compressed.pdf)
