#ifndef _DOCUMENT_STRUCTS_H
#define _DOCUMENT_STRUCTS_H
#include <stdio.h>
//#include <opencv2/opencv.hpp>
//#include <opencv/cv.h>
//#include <opencv2/highgui/highgui.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/ml.h>

using namespace cv;

struct symbol;
struct word;
struct word_line;
struct paragraph;
int rangeOverLap(int x1, int x2, int y1, int y2);
void srcToBinaryInvert(Mat &image, Mat &binary);
void srcToBinary(Mat &image, Mat &binary);
void trainImage(vector<vector<Mat> > &results, Mat &image);
void wordFetch(vector<vector<Mat> > &words, Mat &image, double paragraph_threshold);


struct symbol {
    vector<Point> *points;
    Rect boundingBox;
    Point center;
    
    // The index 
    int paragraph_index;
    char ascii;
    
    explicit symbol(vector<Point> &c);
    void merge(symbol &rhs);
};


struct region {
    
    vector<symbol*> symbols;
    Rect boundingBox;
    
    /*****
    * Adds symbol to list. Prevent empty regions
    *****/
    explicit region(symbol &nSym);

    /*****
    * Adds symbols to the list and resize the bounding box accordingly
    *****/
    virtual void addSymbol(symbol &nSym);
    
    /*****
    * Adds all symbols from rhs to this region's list and resize boundingBox
    *****/
    virtual void merge(region &rhs);
    
};

struct word : public region {
    explicit word(symbol &nSym);
};

struct word_line : public region {
    vector<word> words;
    
    explicit word_line(symbol &nSym);
};

struct paragraph : public region {
    int index;
    Point center;
    vector<word_line> word_lines;
    
    explicit paragraph(int index, symbol &nSym);
    
    /*
    * Add symbol to the current list and change the symbol's paragraph_index
    */
    void addSymbol(symbol &nSym);
    
    /*
    * Add rhs symbol list to current paragraph list and change paragraph_index
    */
    void merge(paragraph &rhs);    
   
    private:
        void calcCenter();
};


// used to sort the boudning boxes of the rectangles

struct symbolSort {
    int scale;
    
    symbolSort(int s) : scale(s) {}
    
    bool operator()(const symbol* a, const symbol* b) {
             
        return ( (a->center.x + scale * a->center.y) < 
                 (b->center.x + scale * b->center.y));
    }
};

struct symbolSortReference {
    int scale;
    
    symbolSortReference(int s) : scale(s) {}
    
    bool operator()(const symbol &a, const symbol &b) {
             
        return ( (a.center.x + scale * a.center.y) < 
                 (b.center.x + scale * b.center.y));
    }
};


struct symbolSortSize {
    int scale;
    
    symbolSortSize(int s) : scale(s) {}
    
    bool operator()(const symbol* a, const symbol* b) {
             
        return ( (a->center.x + scale * a->center.y) < 
                 (b->center.x + scale * b->center.y));
    }
};

struct symbolSortV {
    int scale;
    
    symbolSortV(int s) : scale(s) {}
    
    bool operator()(const symbol* a, const symbol* b) {

        return ( (a->boundingBox.x *scale + a->boundingBox.y) 
                 < (b->boundingBox.x * scale + b->boundingBox.y));
    }
};

struct lineSort {
    int scale;
    
    lineSort(int s) : scale(s) {}
    
    bool operator()(const word_line &a, const word_line &b) {
        return ( (a.boundingBox.x + a.boundingBox.y * scale) < 
                 (b.boundingBox.x + b.boundingBox.y * scale) );
    }
};

/*
struct paragraphSortH {
    int scale;
    
    paragraphSortH(int s) : scale(s) {}
    
    bool operator()(const paragraph &a, const paragraph &b) {
        return ( (a.boundingBox.x + a.boundingBox.y * scale) < 
                 (b.boundingBox.x + b.boundingBox.y * scale) );
    }
};

struct paragraphSortV {
    int scale;
    
    paragraphSortV(int s) : scale(s) {}
    
    bool operator()(const paragraph &a, const paragraph &b) {
        return ( (scale * a.boundingBox.x + a.boundingBox.y ) < 
                 (scale * b.boundingBox.x + b.boundingBox.y ) );
    }
};
*/
struct paragraphSortH {
    int scale;
    
    paragraphSortH(int s) : scale(s) {}
    
    bool operator()(const paragraph &a, const paragraph &b) {
    
        if (rangeOverLap(a.center.x, a.boundingBox.br().x,
                         b.boundingBox.x, b.boundingBox.br().x) > 0) {
            // if two pargraphs overlaps in width range
            return (a.boundingBox.br().y < b.boundingBox.y);
            
        } else if (rangeOverLap(a.boundingBox.y, a.boundingBox.br().y,
                             b.boundingBox.y, b.boundingBox.br().y) > 0) {
                
            return (a.boundingBox.x < b.boundingBox.x);
        } else {
            return ( (a.boundingBox.x + a.boundingBox.y *scale) < 
                 (b.boundingBox.x + b.boundingBox.y *scale ) );
        }
    }
};



#endif
