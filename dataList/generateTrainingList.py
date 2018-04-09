import os

# dataset directory
input_dir = '/Users/yihao/Downloads/HW4/'
# label and image set directory
label_dir =input_dir + 'label_set'
image_dir = input_dir + 'image_set'

list = os.listdir(label_dir)
number_files = len(list)
print(list)
f = open('train.txt','w')
for i in range(number_files):
    f.write('image_set/' + str(i) + '.jpg' + '  ' + 'label_set/' + str(i) + '.png\n')
f.close()

