#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/ml.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include "DocumentStructs.h"
#include "DocumentStructs.cpp"

using namespace cv;
using namespace std;

/*
	processFeatures function process images and derives the features of each image which in our case is a character.
	These features/descriptors of the characters are then added to BOWKMeansTrainer which is a Bag of Features
	This function takes as input parameters a vector of images, this allows us to process all images in one place. 
	We pass by reference our BOWKMeansTrainer (bag of features) since this is a variable we want available in main scope
	A pointer for DescriptorExtractor and SurfFeatureDetector is also passed from main since we define them there already.
	The function process each image by using SurfFeatureDetector to derive keypoints of the image/character.
	Using the pointer DescriptorExtractor extractor we are able to compute features for our character based on the character's keypoints.
	Then, we add the features computed to our bowTrainer (bag of features)
	Although the function is void, we are accessing bowTrainer from main scope so results will be stored there
	Overall, the main purpose of this is to allow us to build a vocabulary of features. 
	However, we wouldn't be able to determine the corresponding character to features
*/
void processFeatures(vector<Mat> images, BOWKMeansTrainer &bowTrainer, Ptr<DescriptorExtractor> extractor, SiftFeatureDetector detector){
	for (int j=0; j<images.size(); j++) {
		Mat image, src;
		
		resize(images.at(j), src, Size(0,0), 10,10);
        
        copyMakeBorder(src, image, 10,10,10,10,BORDER_CONSTANT, Scalar(255));
		

		vector<KeyPoint> keypoints;
		detector.detect(image, keypoints);
		Mat features;
		extractor->compute(image, keypoints, features);
	    bowTrainer.add(features);
		
	}
}

/*
	dataTraining dunction is a function that will allow us to store both character features and their corresponding float value for the ascii chracter they represent. As an example, 'A' is stored as 65.
	The function takes a vector of images, a reference to a Mat array of labels which will hold the float value of ascii characters, a reference to a Mat array of trainingData, SurfFeatureDetector detector and a reference to BOWImgDescriptorExtractor and also the character we are training for.
	We compute the keypoints of a character and their features (bowDescriptor) and store these features in a Mat array called trainingData. We also store the corresponding label for the features as the float value of the ascii character we are seeing.
	The function will access and modify two Mat arrays holding features and the matching labels.
	
*/
void dataTraining (vector<Mat> images, Mat &labels, Mat &trainingData, SiftFeatureDetector detector, BOWImgDescriptorExtractor &bowDE, char character) {
	vector<KeyPoint> keypoints;
	Mat bowDescriptor;
	cout<<images.size()<<" images"<<endl;
	bool verify = false;
	for (int j=0; j<images.size(); j++) {
		Mat image = images.at(j);
		detector.detect(image, keypoints);
		bowDE.compute(image, keypoints, bowDescriptor);
		cout<<keypoints.size()<<" "<<bowDescriptor.size()<<endl;
		if (bowDescriptor.size()!=Size(0,0)) {
			verify=true;
			trainingData.push_back(bowDescriptor);
			labels.push_back((float) character);
		}
	}
	
	cout<<labels.rows<<" "<<trainingData.rows<<endl<<endl;

}

int main( int argc, char** argv ) {
	
	/*
	COMMON VARIABLES
		run:			Determines whether dictionary building or text analysis runs
							Set last argument to 1 or 2
								1) Dictionary Building
								2) Text Analysis
		filepath:	The second to last argument is the folder in which code results will saved to
							dictionaryFilePath & labelsFilePath is saved to filepath
		size:			Size of the image, it is used to resize images for testing & viewing purposes
		threshold:  The threshold at which grayscale becomes a black and white image
		imageCount: The number of images used to train per character, we used three different fonts
		1 - dictionary1 image
		2 - dictionary2 image
		3 - dictionary3 image
		4 - the image to analyze
		5 - filepath to save to so you dictionary.yml, vocab, etc
		6 - 1==dictionary build 2==analysis 
		
	*/
	
	int run = atoi(argv[argc-1]);
	string filepath = argv[argc-2];
	int imageCount = 3;
	string dictionaryFilePath = filepath+"/dictionary.xml";
	string detectedTextFilePath = filepath + "/detectedText.txt";
	string vocabularyFilePath = filepath+"/vocabulary.yml";
	int threshold = 200;
	
	//To the tiny code & set dictionary size
	
	/*
	 
	 Designed to recognize 93 different characters
	
	*/
	int dictionarySize = 93;
	char letters[93];
	char start='!';
	for(int i=0;i<93;i++){
		letters[i]=start;
		start++;
	}
	
	/*
	
	Referenced FoodCamClassifier project at:
		https://github.com/royshil/FoodcamClassifier
	
	*/
	
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int retries = 3;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
	SiftFeatureDetector detector(500);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	
	if (run==1) {
		cout<<"BUILDING DICTIONARY"<<endl;
		
		/*
		
		We iterate a sequence of images and store into a vector before calling processFeatures function to save features into a bag of features
		
		*/
		
		vector< vector<Mat> > trainingImages;
		Mat fileImage1 = imread(argv[1],1);//CV_LOAD_IMAGE_GRAYSCALE);
		Mat dictionary1 = fileImage1>threshold;

		Mat fileImage2 = imread(argv[2],1);//CV_LOAD_IMAGE_GRAYSCALE);
		Mat dictionary2 = fileImage2>threshold;
		
		Mat fileImage3 = imread(argv[3],1);//CV_LOAD_IMAGE_GRAYSCALE);
		Mat dictionary3 = fileImage3>threshold;
		
		
		cout << trainingImages.size() << endl;
		trainImage(trainingImages, dictionary1);
		
		cout << trainingImages.size() << endl;
		trainImage(trainingImages, dictionary2);
		
		cout << trainingImages.size() << endl;
		trainImage(trainingImages, dictionary3);
		cout << trainingImages.size() << endl;
		
		/********** ********** ********** ********** **********
		 
		 I am assuming you are appending the characters hence calling it three times
		 Overall, we expect 93*3=279 items in the vector
		 
		 function(trainingImages, dictionary1);
		 function(trainingImages, dictionary2);
 		 function(trainingImages, dictionary3);
		 Overall
		 
		 If there are errors it's probably the for variable index within the for loop

		 ********** ********** ********** ********** **********/
		
		/* CODEPENDENT CODE GOES HERE */
		for (int i=0; i<dictionarySize; i++) {
            vector<Mat> images;
			for (int j = 0; j<imageCount; j++) {
				int index = i + (dictionarySize*j);
				images.push_back(trainingImages.at(index).at(0));
			}
		    processFeatures(images,bowTrainer,extractor,detector);
		}
		
		
		/*
		
		Based on features stored in bowTrainer, we derive features and cluster the features. This is results in a dictionary/vocabulary of features. 
		We save this to a file for later use when loading SVM
		
		*/
		
		vector<Mat> descriptors = bowTrainer.getDescriptors();
		Mat dictionary = bowTrainer.cluster();
		bowDE.setVocabulary(dictionary);
		FileStorage fs(vocabularyFilePath, FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();
		
		/*
		
		 We will be using SVM (a statistical model) to predict value of future character images.
		 Mat array labels & trainingData is stored using CV_32FC1 format
		 
		*/
		
		Mat labels(0,1,CV_32FC1);
		Mat trainingData(0,dictionarySize,CV_32FC1);
		int charCount=0;
		/*
		 Once again, we read images and call dataTraining function to derive labels and features of each image processed.
		*/
		
		for (int i=0; i<dictionarySize; i++) {
            vector<Mat> images;
			for (int j = 0; j<imageCount; j++) {
				int index = i + (dictionarySize*j);
				images.push_back(trainingImages.at(index).at(0));
			}
		    dataTraining(images, labels, trainingData, detector, bowDE, letters[charCount]);
			charCount++;
		}
		
		
		/*
		
		 Referenced:
			http://answers.opencv.org/question/19062/how-to-train-my-data-only-once/
		
		*/
		
		CvSVMParams params;
		params.kernel_type=CvSVM::RBF;
		params.svm_type=CvSVM::C_SVC;
		params.gamma=0.50625000000000009;
		params.C=312.50000000000000;
		params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
		
		/*
		
		 We train svm (a model with associated learning algorithms for pattern recognition) using the labels & trainingData computed and then save the results into a file
		 This will allow us to load and predict chracters
		
		*/
		CvSVM svm;
		svm.train(trainingData,labels,cv::Mat(),cv::Mat(),params); // BOOOOOOOOOOOOOM
		svm.save(dictionaryFilePath.c_str());
		
	}
	else if (run==2) {
		cout<<"ANALYZING TEXT"<<endl;
		/*
		
		We load svm and vocabulary and set bag of features (bowDE) to contain the features of the stored vocabulary
		We also create a text filr to which we will store the results of our optical character recongition program
		
		*/
		
		ofstream detectedText;
		detectedText.open(detectedTextFilePath.c_str());
		CvSVM svm;
		svm.load(dictionaryFilePath.c_str());
		Mat dictionary;
		FileStorage fs (vocabularyFilePath, FileStorage::READ);
		fs["vocabulary"]>>dictionary;
		bowDE.setVocabulary(dictionary);
		
		/* 
		
		Here we load images corresponding to characters.
			vector <Mat> represents a word formed by a sequence of character images
			vector< vector<Mat> > therefore represents a structure of multiple words (a sentence).
		
		*/
		
		vector < vector<Mat> >  analyzeText;
		Mat fileImage = imread(argv[4],1);
		Mat image = fileImage>threshold;
		wordFetch(analyzeText, image, 150);
		
		/*
		
		 We will analyze the vector of images to derive text based on svm prediction.
		 We iterate each "setence"/sequence of words and then each sequence of images that represent a word. This way for each vector of character images, we form a text word using the character predicted.
		Then, we append to the sentence vector. This will allow us to represent words rather than a block of characters.
		 
		*/
		string word = "";
		vector<string> sentence;
		for (int i=0; i<analyzeText.size(); i++) {
			vector<Mat> wordChars = analyzeText.at(i);
			string word;
			for (int j=0; j<wordChars.size(); j++) {
				vector<KeyPoint> keypoints;
				Mat bowDescriptor;
				Mat image = wordChars.at(j);
				detector.detect(image, keypoints);
				bowDE.compute(image, keypoints, bowDescriptor);
				float response = svm.predict(bowDescriptor);
				word+=(char)response;
			}
			sentence.push_back(word);
		}
		
		/*
		 
		 We have a vector<string> sentence where each string is a word. Puncuation is a part of word stored and we only need to add spaces between strings to form a block of text.
		 Then, we write the results to a text file.
		
		*/
		for (int i=0; i<sentence.size(); i++) {
			detectedText<<sentence.at(i)<<" ";
		}
		detectedText.close();
	}

	return 0;
}
