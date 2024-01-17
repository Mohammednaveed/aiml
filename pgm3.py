import csv
a = []
csvfile = open('pgm3.csv', 'r')
reader = csv.reader(csvfile)
print("Data present in csv file is: ")
for row in reader:
    a.append(row)
    print(row)
num_attributes = len(a[0]) - 1

print("\nInitial hypothesis is ")
s = ['0'] * num_attributes
g = ['?'] * num_attributes
print("The most specific: ", s)
print("The most general: ", g)

for j in range(0, num_attributes):
    s[j] = a[0][j]

print("\nThe candidate algorithm")
temp = []
for i in range(0, len(a)):
    if (a[i][num_attributes] == 'yes'):
        for j in range(0, num_attributes):
            if (a[i][j] != s[j]):
                s[j] = '?'

        for j in range(0, num_attributes):
            for k in range(1, len(temp)):
                if temp[k][j] != '?' and temp[k][j] != s[j]:
                    del temp[k]

        print("\nfor instance {0} the space hypothesis is S{0}\n".format(i + 1), s)
        if (len(temp) == 0):
            print("\nfor instance {0} the hypothesis is G{0}\n".format(i + 1), g)
        else:
            print("\nfor instance {0} the hypothesis is G{0}\n".format(i + 1), temp)

    if (a[i][num_attributes] == 'no'):
        for j in range(0, num_attributes):
            if (a[i][j] != s[j] and s[j] != '?'):
                g[j] = s[j]
                temp.append(g)
                g = ['?'] * num_attributes
                
        print("\nFor instance{0} the hypothesis is s{0}\n".format(i + 1), s)
        print("\nFor instance{0} the hypothesis is g{0}\n".format(i + 1), temp)