#include "DocumentStructs.h"
#include <stdio.h>
//#include <opencv2/opencv.hpp>
#include <fstream>
//#include <opencv/cv.h>
//#include <opencv2/highgui/highgui.hpp>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/ml.h>


using namespace cv;

using std::cout;
using std::endl;


int rangeOverLap(int x1, int x2, int y1, int y2) {
    if (x1 <= y2 && y1 <= x2) {
        if (x2 < y2) {
            return (x2 - y1);
        } else {
            return (y2 - x1);
        }
    }
    return 0;
}

/***********************
* symbol constructor and functions
***********************/

symbol::symbol(vector<Point> &c) {
    points = &c;
    boundingBox = boundingRect(c);
    center.x = boundingBox.x + boundingBox.width / 2;
    center.y = boundingBox.y + boundingBox.height / 2;
    paragraph_index = -1;
    ascii = 0;
}

void symbol::merge(symbol &rhs) {
    Point ttl = boundingBox.tl(),
          tbr = boundingBox.br(),
          rhstl = rhs.boundingBox.tl(),
          rhsbr = rhs.boundingBox.br();

    if (rhstl.x < ttl.x) ttl.x = rhstl.x;
    if (rhstl.y < ttl.y) ttl.y = rhstl.y;
    if (rhsbr.x > tbr.x) tbr.x = rhsbr.x;
    if (rhsbr.y > tbr.y) tbr.y = rhsbr.y;

    boundingBox = Rect(ttl, tbr);
}


/***********************
* region constructor and functions
***********************/
region::region(symbol &nSym) {
    symbols.push_back(&nSym);
    boundingBox = nSym.boundingBox;
}

void region::addSymbol(symbol &nSym) {
    symbols.push_back(&nSym);
    
    Point tl = boundingBox.tl(),
          br = boundingBox.br(), 
          tmp_tl = nSym.boundingBox.tl(),
          tmp_br = nSym.boundingBox.br();
          
    if (tmp_tl.x < tl.x) tl.x = tmp_tl.x;
    if (tmp_tl.y < tl.y) tl.y = tmp_tl.y;
    if (tmp_br.x > br.x) br.x = tmp_br.x;
    if (tmp_br.y > br.y) br.y = tmp_br.y;
    
    boundingBox = Rect(tl,br);
    
}

void region::merge(region &rhs) {
    if(&rhs == this) {
        return;
    }

    symbols.insert( symbols.end(), rhs.symbols.begin(), rhs.symbols.end() );

    Point tl = boundingBox.tl(),
          br = boundingBox.br(), 
          tmp_tl = rhs.boundingBox.tl(),
          tmp_br = rhs.boundingBox.br();
          
    if (tmp_tl.x < tl.x) tl.x = tmp_tl.x;
    if (tmp_tl.y < tl.y) tl.y = tmp_tl.y;
    if (tmp_br.x > br.x) br.x = tmp_br.x;
    if (tmp_br.y > br.y) br.y = tmp_br.y;
    
    
    boundingBox = Rect(tl,br);
}


/***********************
* word constructor and functions
***********************/

word::word(symbol &nSym) : region(nSym) {}

word_line::word_line(symbol &nSym) : region(nSym) {}


/***********************
* paragraph constructor and functions
************************/

paragraph::paragraph(int index, symbol &nSym) : region (nSym) {
    this->index = index;
    nSym.paragraph_index = this->index;
    calcCenter();
}

void paragraph::addSymbol(symbol &nSym){
    symbols.push_back(&nSym);
    nSym.paragraph_index = index;
    
    Point tl = boundingBox.tl(),
          br = boundingBox.br(),
          tmp_tl = nSym.boundingBox.tl(),
          tmp_br = nSym.boundingBox.br();
          
    if (tmp_tl.x < tl.x) tl.x = tmp_tl.x;
    if (tmp_tl.y < tl.y) tl.y = tmp_tl.y;
    if (tmp_br.x > br.x) br.x = tmp_br.x;
    if (tmp_br.y > br.y) br.y = tmp_br.y;
    
    boundingBox = Rect(tl, br);
    calcCenter();
}

void paragraph::merge(paragraph &rhs) {
    if(&rhs == this) {
        return;
    }
    for(int i = 0; i < rhs.symbols.size(); i++) {
        rhs.symbols[i]->paragraph_index = this->index;
    }
    symbols.insert( symbols.end(), rhs.symbols.begin(), rhs.symbols.end() );
    
    Point tl = boundingBox.tl(),
          br = boundingBox.br(), 
          tmp_tl = rhs.boundingBox.tl(),
          tmp_br = rhs.boundingBox.br();
          
    if (tmp_tl.x < tl.x) tl.x = tmp_tl.x;
    if (tmp_tl.y < tl.y) tl.y = tmp_tl.y;
    if (tmp_br.x > br.x) br.x = tmp_br.x;
    if (tmp_br.y > br.y) br.y = tmp_br.y;
    
    boundingBox = Rect(tl,br);
    calcCenter();
}


void paragraph::calcCenter() {
    center.x = boundingBox.x + boundingBox.width / 2.0 + 0.5;
    center.y = boundingBox.y + boundingBox.height / 2.0 + 0.5;
}





/*
* convert image to binary using inverted format:
* background = 0
*/
void srcToBinaryInvert(Mat &image, Mat &binary) {
    Mat gray;
    
    cvtColor(image, gray, CV_RGB2GRAY);

    threshold(gray, binary, 128, 255, THRESH_BINARY_INV);
}

/*
* convert image to binary:
* background = 1;
*/
void srcToBinary(Mat &image, Mat &binary) {
    Mat gray;
    
    cvtColor(image, gray, CV_RGB2GRAY);

    threshold(gray, binary, 128, 255, THRESH_BINARY);
}


/*
*   use only to train images with specific set up
*
*/
void trainImage(vector<vector<Mat> > &results, Mat &image) {
    
    Mat binary;
    srcToBinaryInvert(image, binary);
    
    // find contours in image
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    // find contours of the image 
    findContours( binary, contours, hierarchy, CV_RETR_CCOMP, 
        CV_CHAIN_APPROX_SIMPLE);
    
    srcToBinary(image, binary);
    // get bounding boxes for contours
    vector<symbol> symbols;
    symbols.reserve( contours.size() );
    
    // make found countours whose size is greater than 2 X 2 pixel
    for (int i = 0; i >= 0; i = hierarchy[i][0] ) {
        symbol temp(contours[i]);
        if (temp.boundingBox.width > 2.0 && temp.boundingBox.height > 2.0) {
            symbols.push_back(temp);
        }
    }
    
    vector<symbol*> holder;
    holder.reserve(symbols.size());
   
    if (symbols.size() <= 0) {
        cout << "No symbols could be found!" << endl;
        return;
    }
    
    std::sort(symbols.begin(), symbols.end(),
                  symbolSortReference(std::max(image.rows, image.cols)));
    
    vector<word_line> word_lines(1,word_line(symbols[0]));
    
    int line_index = 0;
    
    for (int j = 1; j < symbols.size(); j++) {
        symbols[j];
        Point tl = word_lines[line_index].boundingBox.tl(),
              br = word_lines[line_index].boundingBox.br();
              
        if ((symbols[j].boundingBox.y >= tl.y && symbols[j].boundingBox.y <= br.y) ||
            (tl.y >= symbols[j].boundingBox.y && tl.y <= symbols[j].boundingBox.br().y)) {
            
            word_lines[line_index].addSymbol(symbols[j]);   
        } else {
            word_lines.push_back(word_line(symbols[j]));
            line_index++;
        }
    }
    
    //short first detected lines based on the top-left corner of the line boundingbox
    std::sort(word_lines.begin(),
              word_lines.end(),
              lineSort( std::max(image.rows, image.cols)));

    for (int j = 0 ; j < word_lines.size()-1; j++) {
                
        Point tl = word_lines[j].boundingBox.tl(),
              br = word_lines[j].boundingBox.br(),
              tl2 = word_lines[j+1].boundingBox.tl(),
              br2 = word_lines[j+1].boundingBox.br();
        
        /* merge lines that overlaps in height */
        if ((tl2.y >= tl.y && tl2.y <= br.y) || 
            (tl.y >= tl2.y && tl.y <= br2.y)) {
            
            word_lines[j].merge(word_lines[j+1]);

            word_lines.erase(word_lines.begin() + j+1);
            j--;
            continue;
        }
    }
    // Line organizing complete
    
    
    // free memory that will no longer be used
    symbols.clear();
    
    /*************************
    *   Word level grouping
    **************************/
    for(int j = 0; j < word_lines.size(); j++) {
        //vector<int> charWordSpacingHistogram(textbox_distance_threshold,0);
        
        int widthSum = 0;

        Point tl = word_lines[j].boundingBox.tl();
        Point br = word_lines[j].boundingBox.br();
        
        /*
        * sort symbols inside lines using the priority:
        * left to right, then top to bottom
        */
        std::sort(word_lines[j].symbols.begin(),
                  word_lines[j].symbols.end(),
                  symbolSortV(std::max(image.rows, image.cols)));
        
        /*******************************
        *   detaching symbol resolving
        ********************************/
        for (int k = 0; k < word_lines[j].symbols.size() - 1; k++) {
            symbol *current = word_lines[j].symbols[k];
            symbol *next = word_lines[j].symbols[k+1];
            
            int difference = rangeOverLap(current->boundingBox.x,
                                    current->boundingBox.br().x,
                    next->boundingBox.x, next->boundingBox.br().x);
            
            if( (double)difference >= (current->boundingBox.width * 0.3) ||
                (double)difference >= (next->boundingBox.width * 0.3)) {
                
                current->merge(*next);
                word_lines[j].symbols.erase(word_lines[j].symbols.begin() + k + 1);
                k--;
                continue;
            }
            widthSum += current->boundingBox.width;
        }
        
        for (int k = 0; k < word_lines[j].symbols.size(); k++) {
            holder.push_back(word_lines[j].symbols[k]);
            
        }

        word_lines[j].symbols.clear();
    }
    if ( matchShapes(*(holder[1]->points),*(holder[2]->points),1, 0.0) 
        < 0.5) {
        holder[1]->merge(*holder[2]);
        holder.erase(holder.begin()+2);
    }
    for (int i = 0; i < holder.size(); i++){
        results.push_back(vector<Mat>(1,Mat(binary, holder[i]->boundingBox)));
    }
}











/*
* param1: fector to hold all mats that are inside a word
* param2: the source RGB image
* param3: user input textbox separating distance threshold:
        used to group symbols together into one textbox
*/
void wordFetch(vector<vector<Mat> > &words, Mat &image, double textbox_distance_threshold)
{ // replace words as necessary

    Mat binary;
    
    if ( !image.data )
    {
        printf("No image data \n");
        return;
    }

    //conver to gray then binary
    //cvtColor(image, gray, CV_RGB2GRAY);
    //threshold(gray, binary, 200, 255, THRESH_BINARY_INV);
    srcToBinaryInvert(image, binary);

    // find contours in image
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    // find contours of the image 
    findContours( binary, contours, hierarchy, CV_RETR_CCOMP, 
        CV_CHAIN_APPROX_SIMPLE);
    
    srcToBinary(image, binary);
    // get bounding boxes for contours
    vector<symbol> symbols;
    symbols.reserve( contours.size() );
    
    // make found countours whose size is greater than 2 X 2 pixel
    for (int i = 0; i >= 0; i = hierarchy[i][0] ) {
        symbol temp(contours[i]);
        if (temp.boundingBox.width > 2.0 && temp.boundingBox.height > 2.0) {
            symbols.push_back(temp);
        }
    }
   
    if (symbols.size() <= 0) {
        cout << "No symbols could be found!" << endl;
        return;
    }


   /*******************************************************************
   * We are finding different paragraphs on the image by comparing each individual
   * symbols' distance away from each other.    
   * Compare every symbol from the bottom right of the page to the top left
   * If the vertical distance between two symbols is less than the user input paragraph
   * threshold then the compared two symbols are inside the same paragraph
   *******************************************************************/
    vector<paragraph> paragraphs;
    paragraphs.reserve(100);
        
    paragraphs.push_back(paragraph(0, symbols[0]));
    
    for (int i = 0; i < symbols.size(); i++) {
        for (int j = 0; j < i; j++) {
            
            // if the vertical distance between i and j is greater than paragraph threshold
            // then all of the symbols above will have distance greater than threshold
            if((symbols[i].center.y - symbols[j].center.y) > textbox_distance_threshold) break;
            
            // Find the distance between symbol i and symbol j
            double diff = cv::norm(symbols[i].center - symbols[j].center);
            
            // If the distance between i and j is less than the paragraph threshold
            // then they belong in the same paragraph
            if(diff < textbox_distance_threshold) {
                if ( symbols[i].paragraph_index < 0) {
                    paragraphs[symbols[j].paragraph_index].addSymbol(symbols[i]);
                } else if (symbols[i].paragraph_index != symbols[j].paragraph_index) {
                    int tempIndex = symbols[j].paragraph_index;
                    paragraphs[symbols[i].paragraph_index].merge(
                                paragraphs[symbols[j].paragraph_index]);
                    paragraphs[tempIndex].index = symbols[i].paragraph_index;
                    paragraphs[tempIndex].symbols.clear();
                }
            }
        }
        if(symbols[i].paragraph_index < 0) {
            int nIndex = paragraphs.size();
            paragraphs.push_back(paragraph(nIndex, symbols[i]));
        }
    }
    
    
    
    // True paragraphs are the root paragraphs
    vector<paragraph> trueParagraphs;    
    for (int i = 0; i < paragraphs.size(); i++) {
        Scalar color(rand()&255,rand()&250,rand()&250);
        if(paragraphs[i].index == i && paragraphs[i].symbols.size() > 0) {
            trueParagraphs.push_back(paragraphs[i]);
        }
    }
    
    
    // free useless paragraphs
    paragraphs.clear();
    /*
    
        organize paragraphs here.
    
    
    */
    std::sort(trueParagraphs.begin(), trueParagraphs.end(),
            paragraphSortH(std::max(image.rows, image.cols)));
    
    
    for (int i = 0; i < trueParagraphs.size(); i++) {
        
        paragraph * currentParagraph = &trueParagraphs[i];
        
        std::sort(currentParagraph->symbols.begin(),
                  currentParagraph->symbols.end(),
                  symbolSort(std::max(image.rows, image.cols)));

        /*************************
        *   Line level grouping
        **************************/
        currentParagraph->word_lines.push_back(word_line(*(currentParagraph->symbols[0])));
        
        int line_index = 0;        
        
        for (int j = 1; j < currentParagraph->symbols.size(); j++) {
            symbol * sym = currentParagraph->symbols[j];
            Point tl = currentParagraph->word_lines[line_index].boundingBox.tl(),
                  br = currentParagraph->word_lines[line_index].boundingBox.br();
                  
            if ((sym->boundingBox.y >= tl.y && sym->boundingBox.y <= br.y) ||
                (tl.y >= sym->boundingBox.y && tl.y <= sym->boundingBox.br().y)) {
                
                currentParagraph->word_lines[line_index].addSymbol(*sym);   
            } else {
                currentParagraph->word_lines.push_back(word_line(*sym));
                line_index++;
            }
        }
        
        //short first detected lines based on the top-left corner of the line boundingbox
        std::sort(currentParagraph->word_lines.begin(),
                  currentParagraph->word_lines.end(),
                  lineSort( std::max(image.rows, image.cols)));

        for (int j = 0 ; j < currentParagraph->word_lines.size()-1; j++) {
            
            word_line * current = &(currentParagraph->word_lines[j]);
            word_line * next = &(currentParagraph->word_lines[j+1]);
            
            Point tl = current->boundingBox.tl(),
                  br = current->boundingBox.br(),
                  tl2 = next->boundingBox.tl(),
                  br2 = next->boundingBox.br();
            
            /* merge lines that overlaps in height */
            if ((tl2.y >= tl.y && tl2.y <= br.y) || 
                (tl.y >= tl2.y && tl.y <= br2.y)) {
                
                current->merge(*(next));

                currentParagraph->word_lines.erase(
                    currentParagraph->word_lines.begin() + j+1);
                j--;
                continue;
            }
        }
        // Line organizing complete
        
        
        // free memory that will no longer be used
        currentParagraph->symbols.clear();
        
        /*************************
        *   Word level grouping
        **************************/
        for(int j = 0; j < currentParagraph->word_lines.size(); j++) {
            //vector<int> charWordSpacingHistogram(textbox_distance_threshold,0);
            
            int widthSum = 0;
            
            word_line *currentLine = &(currentParagraph->word_lines[j]); 

            Point tl = currentLine->boundingBox.tl();
            Point br = currentLine->boundingBox.br();
            
            /*
            * sort symbols inside lines using the priority:
            * left to right, then top to bottom
            */
            std::sort(currentLine->symbols.begin(),
                      currentLine->symbols.end(),
                      symbolSortV(std::max(image.rows, image.cols)));
            
            /*******************************
            *   detaching symbol resolving
            ********************************/
            for (int k = 0; k < currentLine->symbols.size() - 1; k++) {
                symbol *current = currentLine->symbols[k];
                symbol *next = currentLine->symbols[k+1];
                
                int difference = rangeOverLap(current->boundingBox.x,
                                        current->boundingBox.br().x,
                        next->boundingBox.x, next->boundingBox.br().x);
                
                if( (double)difference >= (current->boundingBox.width * 0.3) ||
                    (double)difference >= (next->boundingBox.width * 0.3)) {
                   
                    current->merge(*next);
                    currentLine->symbols.erase(currentLine->symbols.begin() + k + 1);
                    k--;
                    continue;
                }
                widthSum += current->boundingBox.width;
            }
            
            
            /*
            * maxCharSpace: the threshold of the distsance between two symbols'
            *   bounding. If two characters are separated by a distance above this
            *   threshold, the are considered to be part of different words.
            *   This works because the symbols were previous sorted
            */
            
            int avgWidth = (double)widthSum / currentLine->symbols.size();
            int maxCharSpace = avgWidth / 2;

            /**********************
            *   Create word groups
            ***********************/
            currentLine->words.reserve(currentLine->symbols.size() / 5);
            currentLine->words.push_back(word(*(currentLine->symbols[0])));
            int word_index = 0;

            int prevRightEdge = currentLine->symbols[0]->boundingBox.br().x;
            
            Scalar green(20,200,20);
            for (int k = 1; k < currentLine->symbols.size(); k++) {
                
                symbol *current = currentLine->symbols[k];
                
                if ((current->boundingBox.x - prevRightEdge) > maxCharSpace) {
                    // new word
                    currentLine->words.push_back(word(*(current)));
                    words.push_back(vector<Mat>());
                    
                    for (int h = 0; h < currentLine->words[word_index].symbols.size(); h++) {
                        // create Mat for each symbol in sequence and push into result vector
                        
                        words[words.size()-1].push_back(Mat(binary,
                            currentLine->words[word_index].symbols[h]->boundingBox));
                    }
                
                } else {
                    // apend word
                    currentLine->words[word_index].addSymbol(*(current));
                }
                prevRightEdge = current->boundingBox.br().x;
            }
            words.push_back(vector<Mat>());
            for (int h = 0; h < currentLine->words[word_index].symbols.size(); h++) {
                // create Mat for each symbol in sequence and push into result vector
                words[words.size()-1].push_back(Mat(binary,
                    currentLine->words[word_index].symbols[h]->boundingBox));
            }
            // free memory that is no longer useful
            currentLine->symbols.clear();
        }
    }            
}

//int main() {
//    return 0;
//}
