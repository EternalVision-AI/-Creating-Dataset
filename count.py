import os

classes = ["car", "motorcycle", "people", "truck", "bag", "pet", "hat", "camping gear", "bucket", "utv"]
def fileFind():
    files = []
    for i in os.listdir("../Annotations"):
        if i.endswith('.txt'):
            files.append(i)
    # print(files)
    count = [0,0,0,0,0,0,0,0,0,0,0,0]
    for item in files:

        file_data = []
        temp = []

        # print(item)

        with open(item, 'r') as myFile:


            for line in myFile:
                currentLine = line[:-1]
                data = currentLine.split(" ")
                # print(data)
                for i in data:
                    if i.isdigit():
                        count[int(i)] += 1

                        # print(i)
                        temp = float(i)
                        i = str(int(temp))
                        num: int = int(i)
                        # print("item", num)
                        if num > 74:
                            print(item)
    for i in range(0, len(classes)):
        print(classes[i] + " " + str(count[i]))
    
fileFind()