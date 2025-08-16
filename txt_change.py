import os
import glob


def read(directory, height, wide):
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                date1 = line.strip().split(' ')
                type, x_min, y_min, x_max, y_max = int(date1[0]), int(date1[1]), int(date1[2]), int(date1[3]), int(date1[4])
                center_x = ((x_min + x_max) / 2) / wide
                center_y = ((y_min + y_max) / 2) / height
                box_wide = (x_max - x_min)/wide
                box_height = (y_max - y_min)/height
        with open(txt_file, 'w') as f:
            f.write('{} {} {} {} {}'.format(type, center_x, center_y, box_wide, box_height))

read('F:\\gongsi\\auto\\new',497,890)

