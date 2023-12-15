import os


path = '/mnt/workspace/lane_dataset/val'
txt = open(path+os.sep+'list.txt', "w+")
img_list = os.listdir(path)

for img in img_list:
    img_path = path+os.sep+img

    if os.path.isdir(img_path):
        txt.writelines(img_path+'\n')
        print(img_path+'\n')

txt.close()
