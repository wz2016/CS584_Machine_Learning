
file_object = open('car.data','r')

all_the_text = file_object.read();

all_the_text = all_the_text.replace(',', ' ');
# all_the_text = all_the_text.replace('Iris-setosa', '1')
# all_the_text = all_the_text.replace('Iris-versicolor', '2')
# all_the_text = all_the_text.replace('Iris-virginica', '3')
all_the_text = all_the_text.replace('vhigh', '4').replace('high','3').replace('med','2').replace('low','1');
all_the_text = all_the_text.replace('5more','5').replace('more','5');
all_the_text = all_the_text.replace('small','1').replace('med','2').replace('big','3')
all_the_text = all_the_text.replace('vgood','4').replace('unacc', '1').replace('acc','2').replace('good','3')
print len(all_the_text)
file2 = open('car2.txt', 'w')
file2.write(all_the_text);

file_object.close()
file2.close()

