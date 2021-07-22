import pandas as pd 
df = pd.read_csv("./data/AdSmartABdata.csv")

acc = df.shape
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")
