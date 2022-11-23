import csv
file = 'Face recognition attendence\\Datasets\\fetcher.csv' # Target CSV file path
with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
#print(data)
datalist = list(data)


"""version 1"""
#print(datalist[1][1])
# namelist=[]
# for i in range(1,len(datalist)):
#     namelist.append(datalist[i][1])
# print(namelist)

# CrimeCofficient=[]
# for i in range(1,len(datalist)):
#     CrimeCofficient.append(datalist[i][8])
# print(CrimeCofficient)


"""version 2"""
namelist=[]
CrimeCofficient=[]
for i in range(1,len(datalist)):
    namelist.append(datalist[i][1])
    CrimeCofficient.append(float(datalist[i][8]))
# print(namelist)
# print(CrimeCofficient)


"""version 3"""
fetcher = {}
# Adding elements one at a time
for i in range(len(namelist)):
    fetcher[namelist[i]] = CrimeCofficient[i]
#print(fetcher)
print(fetcher['Carl Eugene Watts']+1.00)