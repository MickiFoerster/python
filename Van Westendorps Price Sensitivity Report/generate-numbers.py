import random


#num_of_values = 1428
num_of_values = 10

def generate(von, bis):
    global num_of_values

    delta = bis - von
    middle = delta / 2.

    mean = von + middle
    stdev = middle/3   # 99.73% chance the sample will fall in your desired range

    values = []
    while len(values) < num_of_values:
        sample = random.gauss(mean, stdev)
        if sample >= von and sample <= bis:
            values.append(sample)

    return values

zu_billig = generate(0.75, 1.28)
billig = generate(0.81, 1.39)
teuer = generate(0.89, 1.32)
zu_teuer = generate(1.06, 1.50)

def export(zu_billig, billig, teuer, zu_teuer):
    print("\"zu billig\"; \"billig\"; \"teuer\"; \"zu teuer\";")

    for i in range(num_of_values):
        print("{:.2f}; {:.2f}; {:.2f}; {:.2f};"
              .format(zu_billig[i], billig[i], teuer[i], zu_teuer[i]))


#export(zu_billig, billig, teuer, zu_teuer)

zu_billig.sort(reverse=True)
billig.sort(reverse=True)
teuer.sort()
zu_teuer.sort()
print(zu_billig)
print(billig)
print(teuer)
print(zu_teuer)

for i in range(num_of_values):
    print("{}    {}".format(((i+1)*100)/num_of_values, zu_billig[i]))
