# **Finding Lane Lines on the Road with Matlab** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is an attempt for ‘Udacity - Self-Driving Car NanoDegree: Finding Lane Lines on the Road’

<img src="Project1_ImageOnly\test_images_output\Out1.JPG" width="480" alt="Image Only" />

##Major differences
- I use MATLAB to process matrix operations. 
I am new to Python. Python cannot directly calculate matrix but with ‘numpy’. It seems not convenient. So, I pass the message to MATLAB. Do not try to convert ‘nparray’ with ‘Matlab Array’. It is sadly slow!

- I use OpenCV to access frames in the movie.
The Udacity suggests ‘moviepy’; however, it should install with ‘pip’, which will break the installation of Anaconda. The OpenCV reads a frame with BGR. Convert it to RGB. The process of installing OpenCV from ‘conda’ is very slow in China. It has to be downed manually with ‘conda install --use-local FileName.bz2’ 

- I separate the figure for left and right eyes.
The human has two eyes, So I separate a figure into left and right parts. The two lines on the road are naturally separated. 

- I use a fitting method in MATLAB to find one line.
The teacher of Udacity suggest turning parameters. But, I use Hough algorithm to find many short lines. Then, the starts and ends of lines are grouped together. A fitting method is used to find a line which tries to pass these points. In this way, some error lines are filtered. This method may find these curve lines on the road. 

- I filter error lines with the gradient.
The Hough algorithm usually finds many lines. Some of them are noise with error gradient. I filter the lines with ‘abs(gradient)’ smaller than 15deg and bigger than 75deg. 

- I don’t draw lines which starts from the edge of interest range.
The reason is there is no line. The code can be fixed if you want this function.

##Existing Problems 
- It cannot process the condition that no line has been detected.
- It cannot process the condition that the car is off the traffic lane. 
One line should be in the left part of the figure, and another one in the right figure.
- It will be infected by ‘smart errors’, e.g. black car with white wheels.


##Further improvements
The clustering should be used to separate two lanes together with my  'left and right eyes'.


------------


Blue Bird
Best Wishes