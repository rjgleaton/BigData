import os, numpy, pdb, csv
import pandas as pd


def main():
    part1()
    part3()


def part1():
    filepath = 'C:\\Users\\rjgle\\Documents\\CSCE 587\\homework1\\BBC-textdata\\bbc'

    #Prepare for writing to csv file at end
    file_write = "report.csv"
    fields = ['Article-ID', 'No. Words', 'No. Paragraphs']
    rows = []

    for directory_name in os.listdir(filepath):

        #Ignore readme file
        if(directory_name=="README.TXT"):
            continue

        new_filepath = filepath+'\\'+directory_name
        for filename in os.listdir(new_filepath):
            text_file_path = new_filepath+'\\'+filename
            
            file = open(text_file_path)
            data = file.read()
            words = data.split()
            paragraph = data.split('\n')

            #Delete empty paragraphs
            while("" in paragraph):
                paragraph.remove("")
            
            rows.append([directory_name+'_'+filename, len(words), len(paragraph)])
    
    #Write output to csv file
    with open(file_write, 'w', newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def part3():
    filepath = "C:\\Users\\rjgle\\Documents\\CSCE 587\\homework1\\Absenteeism_at_work.csv"
    #Step 1
    df = pd.read_csv(filepath, delimiter=";")

    #Step 2
    df = df.sample(frac=1)
    print(df.describe())
    #Step 3, 4, 5:
    train = df.head(int(len(df)*0.7))
    valid = df.iloc[int(len(df)*0.7):int(len(df)*0.85)]
    test = df.iloc[int(len(df)*0.85):]
    
    train.to_pickle("train.pickle")
    valid.to_pickle("valid.pickle")
    test.to_pickle("test.pickle")

    #Step 6:
    df_small = df.iloc[:,:5]
    train_small = df_small.head(int(len(df)*0.7))
    valid_small = df_small.iloc[int(len(df)*0.7):int(len(df)*0.85)]
    test_small = df_small.iloc[int(len(df)*0.85):]

    train_small.to_pickle("train_small.pickle")
    valid_small.to_pickle("valid_small.pickle")
    test_small.to_pickle("test_small.pickle")


if __name__ == "__main__":
    main()