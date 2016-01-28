Anahi Garnelo, Shankun Lin, Rebeca Otero, Ying Fang Lee


meraocr:
	Code that recognizes text from images. This code is not currently complete. It is meant
	to run in 2 steps. The program compiles and the first (character learning) executes but
	the second (character matching) step does not execute. The problem seems to lie in some
	errors from the database created in the charcter learning that do not allow the character
	matching to run properly as intended.
	Utilizes:
	DiplayImage.cpp
	
DisplayImage:

Known bugs/issues/flaws:
In DocumentStructs::wordFetch:
    Each textbox may not be organized in its logical readering order. This 
    is a flaw in our sorting algorithmn for the textboxes.

"DocumentStructs.h"
To determine the position of each individual characters on the imnage
several strcuts were defined to organize the layout of image:
	symbol: 
		Each symbol contains a vector of points which is the countour
		of the page and its corresponding bounding box with the center
		of the box recorded and belongs to a pragraph
		
		* Variables associated with symbols:
		vector<Point> *points - the countour of each symbol
		Rect boundingBox - the bounding Rectangle of each symbol countour
		Point center - the center of the boundingBox
		int paragraph_index - the paragraph each symbol belongs to
		
		* Functions associated with symbols:
		
		- symbol(vector<Point> &c)
			The constructor takes in a vector of contours by reference
			and construct the bounding rectangle then calculate the
			center of the boundingBox. Paragraph_index is initialized
			to -1 until later update.
		- merge(symbol &rhs)
			The merge function will compare two symbols that are next
			to each other and combine the respective bounding boxes of
			each symbol.
	
	region: 
		It is the base class for words, lines and paragraphs, contains
		the fundamental parts of every region of text
		
		* Variables associated with region:
		vector<symbol*> symbols - vector containing all of the symbols
			within a region of text
		boindingBox - the rectangle that encloses a region of text
		
		* Functions associated with region:
		- region(symbol & nSym)
			each region will have at least one symbol with its bounding
			box. 
		- addSymbol (symbol &nSym) 
      adds the individual symbols to its vector of symbols and 
      combine the symbols bounding boxes
    - merge(region &rhs)
      Combining two regions together and merge the bounding boxes
      of each region
		
		word:
			A region of symbols tht defines a word
    
    word_line:
			A region of symbols that has a vector of words which defines
      a line
    
    paragraph:
		  A region that contains a group of word_lines that make up a
      paragraph
        
      * Variables associated with paragraph:
			int index - the index of the paragraph
      Point center - the center of the region that contains the 
        paragraph
      vector<word_line> word_lines - all the lines that makes up
        a paragraph
      
      * Functions associated with paragraph:
	  - paragraph(int indes, symbol &nSym)
        make a new paragraph with index and update the pargraph
        index of symbol then calculated the paragraphs center
      - addSymbol(symbol &nSym)
		adds the symbols of the regeion and calculate the respective
		bounding box 
	  - merge(paragraph &rhs)
		merges the same paragraphs and calculate its new bounding box
	
	  symbolSort:
      - sorts the symbols base on its coordinates on the
        there is sorting priorities on the y coordinates
      
      symbolSortReference:
	  - sorts the initial vectors of symbols as we found them
      
      symbolSortV:
	  - sorts the symbols from left to right and bottom up
      
      paragraphSortH:
	  - sorts the paragraphs by its positons
      
      lineSort:
	  - sorts the lines in the paragraphs by its position

learnmatch:
	This code runs in two different settings: character learning and character matching. In
	character learning it takes 3 images (1 character in 3 different fonts) of several different 
	characters, calculates the keypoints of the characters, and saves the results in a database
	file. The character matching takes in images of individual characters, computes the keypoints
	for them, and matches them with the characters in the learned characters database file.


--------------------		
To compile in Linux:
--------------------

make all

-------
To Run:
-------

meraocr (Character Learn)=
./DisplayImage {alphabet1.jpg} {alphabet2.jpg} {alphabet3.jpg} {analyze.jpg} {file path} 1

meraocr (Character Match)=
./DisplayImage {alphabet1.jpg} {alphabet2.jpg} {alphabet3.jpg} {analyze.jpg} {file path} 2

DiplayImage (Stand alone LayoutAnalysisStandalon)=
./LayoutAnalysisStandalon inputImage threshold

learnmatch (Character Learn-stand alone)=
./CharMatch m1.jgp m2.jpg m3.jgp e1.jgp e2.jpg e3.jgp r1.jgp r2.jpg r3.jgp a1.jgp a2.jpg a3.jgp {file path} 1

learnmatch (Character Match- stand alone)=
./CharMatch m1.jgp m2.jpg m3.jgp e1.jgp e2.jpg e3.jgp r1.jgp r2.jpg r3.jgp a1.jgp a2.jpg a3.jgp {file path} 2
