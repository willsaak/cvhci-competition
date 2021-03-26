# 2018/2019 CVHCI Competition
This is the repo for the competition [2018/2019 CVHCI](http://141.3.14.114/judge19/). There are my solutions for all three assignments. These solutions were the best ones, as one can see in the ranking (username: fanta3). More details about the assigmnents could be found on official webpage.

To run this project:
1) Download or clone this repo
2) In terminal go to repo directory and run `direnv allow` and wait till all requirements are installed.
3) Scripts for all the assignments can be found in `main` folder.
4) The datasets can be found on official webpage.
5) Add every dataset folder to `data/data_a{assignment_number}`.

P.S. I use also some other datasets and other tricks, thats why you can have the score lower then my on competition web page.

## 1st Assignment: Color-based skin classifier
This assignment consists on developing a color-based skin classifier. Please explore the provided code for details. You should only modify the code from skinmodel.cc.

The challenge is scored using the F1-measure. ROC graphs are shown for comparison purposes only.

## 2nd Assignment: Person detector
For this assignment you should create a person detector. You will be supplied with several 96x160px image patches as a train set, the positive samples include a person centered and sized 64x128px, therefore a little bit of background is always included. The test samples are 70x136px.

The challenge is scored using the F1-measure. ROC graphs are shown for comparison purposes only.

## 3rd Assignment: Train a FACE Similitude Measure
data: folder with the public data set (853 pairs of faces 250x250px, eye-aligned)

The goal is to train a classifier able to discern if two face images belong to the same person, or belong to different persons.
The main program uses a train and validation setup where half of the image pairs will belong to the same person and half will belong to different persons.
First you should train your similitude measure, and then you will be asked to give a similitude value for several pairs of images.
The score scale is not important, but must give larger values for pairs of faces belonging to the same person than for pairs of faces belonging to different persons.
