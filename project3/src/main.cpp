#include <iostream>
#include <vector>
#include <map>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#include <math.h>
#include <limits>
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
        countImagesforEachDigit[static_cast<int>(trainLabels[i])]++;
    }
//        for (int i = 0; i < 10; i++){
//            std::cout << i << "  " << countImagesforEachDigit[i] << std::endl;
//        }
        //50980 + 1135 + 1032 + 1010 + 982 + 892 + 958 + 1028 + 974 + 1009
    double plClassPixel2DArray [10][784];
    //Initialize the 2d array to count white pixels
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 784; j++){
            plClassPixel2DArray [i][j] = 0;
        }
    }
    //Fill the 2d array by the count of white pixels of each class
    //Step 0 Iterate over the images and count the number of white pixels
    //Not range based for loop cuz order matters
    for (int imageIndex = 0; imageIndex < 60000; imageIndex++){
        //Identify the actual number of this image
        int digitClass = static_cast<int>(trainLabels[imageIndex]);
        for (int pixelIndex = 0; pixelIndex < 784; pixelIndex++){
            int pixelIndexValue = static_cast<int>(trainImages[imageIndex][pixelIndex]);
            if (pixelIndexValue == 1){
                plClassPixel2DArray[digitClass][pixelIndex]++;
                //It means for current digit e.g. 3
                //there is one image whose actual number is this current digit
                //and this image's [pixelIndex] pixel is white
            }
        }
    }
    //Step 1 Calculate the probability PL(Fj = 1|C = c)
    for (int classIndex = 0; classIndex < 10; classIndex++){
        int countOfDigitC = countImagesforEachDigit[classIndex] + 2;//Denominator
        for (int pixelIndex = 0; pixelIndex < 784; pixelIndex++){
            //Nominator
            int countImagesOfDigitCWherePixelFjIsWhite = plClassPixel2DArray[classIndex][pixelIndex] + 1;
            //Fill the 2d array with PL(Fj = 1|C = c) instead
            plClassPixel2DArray[classIndex][pixelIndex]
            = countImagesOfDigitCWherePixelFjIsWhite * 1.0 / countOfDigitC;
        }
    }
//    //Step 2 Calcualte P(Fj = 0|C = c), which is ( 1-PL(Fj = 1|C = c) )
//    double plClassPixel2DArrayFjIs0 [10][784];
//    //Initialize the 2d array
//    for (int i = 0; i < 10; i++){
//        for (int j = 0; j < 784; j++){
//            plClassPixel2DArrayFjIs0 [i][j] = 1 - plClassPixel2DArray[i][j];
//        }
//    }
//    Test try to print the probability image that pixel is white for class = 3
//    for (int f=0; f<numFeatures; f++) {
//        double pixelDoubleValue = static_cast<double>(plClassPixel2DArray[0][f]);
//        if (f % width == 0) {
//            std::cout<<std::endl;
//        }
//        std::cout<<pixelDoubleValue<<" ";
//    }
    //Output 10 bitmaps
    for (int c = 0; c < numLabels; c++) {
        std::vector<unsigned char> classFs(numFeatures);
        for (int f=0; f<numFeatures; f++) {
            //TODO: get probability of pixel f being white given class c
            double p = plClassPixel2DArray[c][f];
            uint8_t v = 255*p;
            classFs[f] = (unsigned char)v;
        }
        std::stringstream ss;
        ss << "../output/digit" <<c<<".bmp";
        Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
    }
    //Output network.txt
    std::ofstream myfile;
    myfile.open("../output/network.txt");
    //The first 784 lines should be P(Fj = 1|C = 0)
    for(int i = 0; i < 784; i++){
        myfile << plClassPixel2DArray[0][i] << std::endl;
    }
    //The next 784 lines should be P(Fj = 1|C = 1)
    for(int i = 0; i < 784; i++){
        myfile << plClassPixel2DArray[1][i] << std::endl;
    }
    //prior probabilities for each class c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    for (int i = 0; i < 10; i++){
        myfile << countImagesforEachDigit[i] * 1.0 / 60000 << std::endl;
    }
    myfile.close();
    //Output classification-summary.txt
    int classificationMatrix [10][10];
    //Initialize the matrix
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            classificationMatrix[i][j] = 0;
        }
    }
    //Iterate over the testing set
    for (int imageIndex = 0; imageIndex < 10000; imageIndex++){
        int currImageAns = static_cast<int>(testLabels[imageIndex]);
        int currImageEval = -1;
        double probSumMax = std::numeric_limits<double>::lowest();;
        //Iterate over 0-9. Find the label with highest probability
        for (int label = 0; label < 10; label++){
            double probSum = 0;
            //Iterate over the image pixels
            for (int pixelIndex = 0; pixelIndex < 784; pixelIndex++){
                int pixelIntValue = static_cast<int>(testImages[imageIndex][pixelIndex]);
                if (pixelIntValue == 1){
                    probSum += log2(plClassPixel2DArray[label][pixelIndex]);
                }
                else if (pixelIntValue == 0){
                    probSum += log2(1 - plClassPixel2DArray[label][pixelIndex]);
                }
            }
            probSum += log2( countImagesforEachDigit[label] * 1.0 / 60000 ) ;
            if (probSum > probSumMax){
                probSumMax = probSum;
                currImageEval = label;
            }
        }
        classificationMatrix[currImageAns][currImageEval]++;
    }
    //Now output
    std::ofstream classificationFile;
    classificationFile.open("../output/classification-summary.txt");
    int correctCount = 0;
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            classificationFile <<classificationMatrix[i][j] << " ";
            if (i == j){
                correctCount += classificationMatrix[i][j];
            }
        }
        classificationFile << std::endl;
    }
    classificationFile << (correctCount * 1.0 / 10000) * 100 << "%";
    classificationFile.close();
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
