# -*- coding: utf-8 -*-

import os
import os.path


def rename():
    path = "E:\\master\\DATA\\GPS_MIT\\2013\\figure20130317\\"

    file_list = os.listdir(path)
    count = 0
    for file in file_list:

        old_dir = os.path.join(path, file)
        if os.path.isdir(old_dir):
            continue
        file_name = os.path.splitext(file)[0]
        file_type = os.path.splitext(file)[1]

        # new_dir = os.path.join(path, "image%03d" % count + ".png")
        # os.rename(old_dir, new_dir)
        count += 1
        if count <= 10:
            print(old_dir)


rename()
print("work done")
