GSR Data Collection Code:

There are two parts to this:  
**1.)** firmware for the arduino: `IMAGS_GSR.ino`  
**2.)** Python script that uses SerialCup (https://github.com/sensortea/SerialCup) to read the data and save it as an csv  

to use the python script, you must open it and adjust two things:  

**1.)** the file path where you want the data to be saved, called "data_path"  
**2.)** the serial number of your arduino. This can be found using the serialcup command "list", executed as `python3 serialcup.py list`  

**Note:** serialCup creates a new .txt file with the data gathered for every test you run. If you keep the file name of that txt as default, when you go to run it again serialcup will make a copy of that original file and then append your new data to it, which will result in erroneous data for the time period between when you finished the previous test and began the next. To avoid this, rename the .txt file after each test.  

As of now I'm using a seperate matlab script to graph the csv data (not shared), but once I get around to learning how matplotlib works I'll integrate that instead.




