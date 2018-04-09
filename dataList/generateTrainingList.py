import os

# dataset directory
input_dir = '/home/yihao/Documents/dataset/tusimple/'
# label and image set directory
label_dir =input_dir + 'label_set_new'
image_dir = input_dir + 'train_set_new'

list = os.listdir(label_dir)
number_files = len(list)
print(list)
f = open('train.txt','w')
for i in range(1,number_files+1):
    f.write('image_set/' + str(i) + '.jpg' + '  ' + 'label_set/' + str(i) + '.png\n')
f.close()