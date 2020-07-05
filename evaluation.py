from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

Image(filename="evaluation/fold_1/validation.dice_soft.png")
Image(filename="evaluation/fold_1/validation.loss.png")
Image(filename="evaluation/fold_1/validation.dice_crossentropy.png")

kidney = []
tumor = []
start = True
with open("evaluation/fold_0/detailed_validation.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        print(line)
        if start:
            start = False
            continue
        kidney.append(float(line[2]))
        tumor.append(float(line[3]))


kidney = np.asarray(kidney)
tumor = np.asarray(tumor)

fig, ax = plt.subplots()
ax.set_title('Evaluation plot for KiTS19')
ax.set_ylabel('Dice Similarity Coefficient')
ax.set_xticklabels(["Kidney", "Tumor"])
ax.boxplot([kidney, tumor])
plt.show()

print("Mean Kidney: " + str(np.mean(kidney)))
print("Mean Tumor: " + str(np.mean(tumor)))

print("Median Kidney: " + str(np.median(kidney)))
print("Median Tumor: " + str(np.median(tumor)))