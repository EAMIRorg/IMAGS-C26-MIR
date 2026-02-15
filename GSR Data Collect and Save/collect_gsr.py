import datetime
import serialcup
import pathlib
import csv

# add your correct data path here
data_path = pathlib.Path("~/Desktop/imags/data").expanduser()
#print(data_path)

# add serial number of specific device. can find s/n by running:
# "python3 serialcup.py list"
serialNum = "03536373332351419152"
baud = 9600
start_flag = "start"
stop_flag = "stop"


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    

    try:
        print("> Type 'start' to begin logging data and 'stop' to end")
        print("> Type CTRL+C to finish")
        serialcup.capture_serial_data(str(data_path), serialNum, baud)
    except KeyboardInterrupt:
        print("\n All Done")

    results = []
    
    def collect_data(line):
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    results.append([int(parts[0]), float(parts[1])])
                except ValueError:
                    pass

    serialcup.query(str(data_path), serialNum, start_flag, stop_flag, collect_data)

    if results:
        csv_name = data_path / f"gsr_results_{timestamp}.csv"

        with open(csv_name,'w') as csvfile:
            writeCSV = csv.writer(csvfile)
            writeCSV.writerow(['ms_timestamp', 'GSR', 'timestamp'])
            for ms_ts, gsr_val in results:
                row_ts = datetime.datetime.fromtimestamp(ms_ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                writeCSV.writerow([ms_ts, gsr_val, row_ts])

        print(f" Saved to\n  {csv_name}")
        print("> Be sure to rename .txt to avoid appending new data on previous test")
    else:
        print("no data found")
if __name__=="__main__":
    main()

