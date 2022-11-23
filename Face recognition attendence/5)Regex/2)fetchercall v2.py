"""this version is for clean code"""
import csv
file = 'Face recognition attendence\\Datasets\\fetcher.csv' # Target CSV file path
with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
datalist = list(data)
namelist=[]
CrimeCofficient=[]
for i in range(1,len(datalist)):
    namelist.append(datalist[i][1])
    CrimeCofficient.append(float(datalist[i][8]))
fetcher = {}
for i in range(len(namelist)):
    fetcher[namelist[i]] = CrimeCofficient[i]
print(fetcher['Carl Eugene Watts']+1.00)