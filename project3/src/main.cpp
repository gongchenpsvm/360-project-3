#include <iostream>
#include <vector>
#include <map>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#define MNIST_DATA_DIR "../mnist_data"
int main(int argc, char* argv[]) {
    //Read in the data set from the files
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
    //Binarize the data set (so that pixels have values of either 0 or 1)
    mnist::binarize_dataset(dataset);
    //There are ten possible digits 0-9 (classes)
    int numLabels = 10;
    //There are 784 features (one per pixel in a 28x28 image)
    int numFeatures = 784;
    //Each pixel value can take on the value 0 or 1
    int numFeatureValues = 2;
    //image width
    int width = 28;
    //image height
    int height = 28;
    //image to print (these two images were randomly selected by me with no particular preference)
    int trainImageToPrint = 50;
    int testImageToPrint = 5434;
    // get training images
    std::vector<std::vector<unsigned char>> trainImages = dataset.training_images;
    // get training labels
    std::vector<unsigned char> trainLabels = dataset.training_labels;
    // get test images
    std::vector<std::vector<unsigned char>> testImages = dataset.test_images;
    // get test labels
    std::vector<unsigned char> testLabels = dataset.test_labels;
    
    //To learn the naive Bayesian classifier,
    //you should do all the following steps only for the training set.
    int countImagesforEachDigit [] = {0,0,0,0,0,0,0,0,0,0};
    for (int i  = 0; i < trainLabels.size(); i++){
        countImagesforEachDigit[static_cast<int>(testLabels[i])]++;
    }
//        for (int i = 0; i < 10; i++){
//            std::cout << i << "  " << countImagesforEachDigit[i] << std::endl;
//        }
        //50980 + 1135 + 1032 + 1010 + 982 + 892 + 958 + 1028 + 974 + 1009
    double plClassPixel2DArray [10][783];
    //Initialize the 2d array
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 783; j++){
            plClassPixel2DArray [i][j] = 0;
        }
    }
    //Fill the 2d array by the count of white pixels of each class
    //Step 0 Iterate over the images and count the number of white pixels
    //Not range based for loop cuz order matters
    for (int imageIndex = 0; imageIndex < 600000; imageIndex++){
        int digitClass = static_cast<int>(trainLabels[imageIndex]);
        std::vector<unsigned char> currImage = trainImages[imageIndex];
        for (int pixelIndex = 0; pixelIndex < 783; pixelIndex++){
            int pixelIndexValue = static_cast<int>(currImage[pixelIndex]);
            if (pixelIndexValue == 1){
                plClassPixel2DArray[digitClass][pixelIndex]++;
            }
        }
    }
    //Step 1 Calculate the probability PL(Fj = 1|C = c)
    for (int classIndex = 0; classIndex < 10; classIndex++){
        int countOfDigitC = countImagesforEachDigit[classIndex] + 2;//Denominator
        for (int pixelIndex = 0; pixelIndex < 783; pixelIndex++){
            //Nominator
            int countImagesOfDigitCWherePixelFjIsWhite = plClassPixel2DArray[classIndex][pixelIndex] + 1;
            plClassPixel2DArray[classIndex][pixelIndex]
            = countImagesOfDigitCWherePixelFjIsWhite * 1.0 / countOfDigitC;//PL(Fj = 1|C = c)
        }
    }
    //Step 2 Calcualte P(Fj = 0|C = c)
    double plClassPixel2DArrayFjIs0 [10][783];
    //Initialize the 2d array
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 783; j++){
            plClassPixel2DArrayFjIs0 [i][j] = 1 - plClassPixel2DArray[i][j];
        }
    }
    
    
//    //print out one of the training images
//    for (int f=0; f<numFeatures; f++) {
//        // get value of pixel f (0 or 1) from training image trainImageToPrint
//        int pixelIntValue = static_cast<int>(trainImages[trainImageToPrint][f]);
//        if (f % width == 0) {
//            std::cout<<std::endl;
//        }
//        std::cout<<pixelIntValue<<" ";
//    }
//    std::cout<<std::endl;
//    // print the associated label (correct digit) for training image trainImageToPrint
//    std::cout<<"Label: "<<static_cast<int>(trainLabels[trainImageToPrint])<<std::endl;
//    //print out one of the test images
//    for (int f=0; f<numFeatures; f++) {
//        // get value of pixel f (0 or 1) from training image trainImageToPrint
//        int pixelIntValue = static_cast<int>(testImages[testImageToPrint][f]);
//        if (f % width == 0) {
//            std::cout<<std::endl;
//        }
//        std::cout<<pixelIntValue<<" ";
//    }
//    std::cout<<std::endl;
//    // print the associated label (correct digit) for test image testImageToPrint
//    std::cout<<"Label: "<<static_cast<int>(testLabels[testImageToPrint])<<std::endl;
//    std::vector<unsigned char> trainI(numFeatures);
//    std::vector<unsigned char> testI(numFeatures);
//    for (int f=0; f<numFeatures; f++) {
//        int trainV = 255*(static_cast<int>(trainImages[trainImageToPrint][f]));
//        int testV = 255*(static_cast<int>(testImages[testImageToPrint][f]));
//        trainI[f] = static_cast<unsigned char>(trainV);
//        testI[f] = static_cast<unsigned char>(testV);
//    }
//    std::stringstream ssTrain;
//    std::stringstream ssTest;
//    ssTrain << "../output/train" <<trainImageToPrint<<"Label"<<static_cast<int>(trainLabels[trainImageToPrint])<<".bmp";
//    ssTest << "../output/test" <<testImageToPrint<<"Label"<<static_cast<int>(testLabels[testImageToPrint])<<".bmp";
//    Bitmap::writeBitmap(trainI, 28, 28, ssTrain.str(), false);
//    Bitmap::writeBitmap(testI, 28, 28, ssTest.str(), false);
    return 0;
}

