GSR Data Collection Code:

There are two parts to this:
1.) firmware for the arduino. this can be found at https://github.com/EAMIRorg/GSR/blob/cb37efe34f5903da14e9539d9df281612ad0b46d/Arduino/IMAGS_GSR.ino
2.) Python script that uses SerialCup (https://github.com/sensortea/SerialCup) to read the data and save it as an csv

to use the python script, you must open it and adjust two things:

1.) the file path where you want the data to be saved, called "data_path"
2.) the serial number of your arduino. This can be found using the serialcup command "list", executed as `python3 serialcup.py list`

Note: serialCup creates a new .txt file with the data gathered for every test you run. If you keep the file name of that txt as default, when you go to run it again serialcup will
make a copy of that original file and then append your new data to it, which will result in erroneous data for the time period between when you finished the previous test and began
the next. To avoid this, rename the .txt file after each test.




