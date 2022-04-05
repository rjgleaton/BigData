import pandas as pd
import csv, pdb

def main():
    filepath = filepath = "C:\\Users\\rjgle\\Documents\\CSCE 587\\homework1\\housing.csv"
    df=pd.read_csv(filepath)
    
    data = df[['median_house_value','ocean_proximity']]

    data = data.groupby('ocean_proximity').agg('mean').sort_values('median_house_value',ascending=False)

    data.to_csv('housing_pandas.csv')



if __name__ == "__main__":
    main()