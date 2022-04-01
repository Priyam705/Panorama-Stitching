# Panorama-Stitching

-Following is the description of all the files present:
	-main.py is the main, executing it will save the final output
	-harris_17.py contains the definition of harris corner detector.

 
-Following should be the structure of directory:
	-The input images of same scene should be stored in the single directory and that directory should be stored in directory main.
	
-Steps to run the file:
	1) Open either linux terminal or anaconda terminal window.
	2) Make your current directory as main.
	3) Type the following command
		python main.py --path <folder name without quotes, where input images are stored>
	4) The final output will be stored as "foldername_output.png" inside the main folder
