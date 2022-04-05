import numpy as np
import csv, pdb

def main():
    filepath = "C:\\Users\\rjgle\\Documents\\CSCE 587\\homework1\\housing.csv"
    data = np.loadtxt(filepath, dtype = {'names':('median_house','ocean_prox'),
            'formats': ('i4','S16')}, delimiter=',', skiprows=1,usecols=(8,9))
    
    ocean_prox = np.unique(data['ocean_prox'])

    prox_avg_list = []
    for prox in ocean_prox:
        prox_avg_list.append([prox,data[data['ocean_prox']==prox]['median_house'].mean()])
    
    array = np.asarray(prox_avg_list,dtype=str)
    array = array[np.argsort(array[:,1])][::-1]

    #Headers for csv file
    headers = ['ocean_proximity', 'median_house_value']
    #filename to save to
    file_writer = 'housing_numpy.csv'



    with open(file_writer,'w',newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(headers)
        csvwriter.writerows(array)

    pdb.set_trace()

if __name__ == "__main__":
    main()