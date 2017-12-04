# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os

sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn

import subprocess
import numpy as np

data_root_path = '/home/cdpt/dataset/KITTI'

classes=['car']

#可以看得出用的是相对路径，因为调用的时候是`python examples/detector.py`
net = dn.load_net("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc_final.weights", 0)
meta = dn.load_meta("cfg/voc.data")
image_set_file = "data/voc/2007_test.txt" 


# load all boxes
with open(image_set_file) as f:
    image_idx=[x.strip() for x in f.readlines()]

num_images=len(image_idx)

all_boxes=[[[] for _ in xrange(num_images)] for _ in xrange(1)] #only car

for i in xrange(num_images):
    r = dn.detect(net, meta, image_idx[i])
    for j in xrange(len(r)):
        all_boxes[0][i].append(r[j])

#evaluate detection
eval_tool = './src/kitti-eval/cpp/evaluate_object'
eval_dir = "results"
det_file_dir = os.path.join(eval_dir, 'data')

if not os.path.isdir(det_file_dir):
    os.makedirs(det_file_dir)

for im_idx, index in enumerate(image_idx):

    filename_pre = index.split('/')[-1].split('.')[0]
    filename = os.path.join(det_file_dir, filename_pre+'.txt')
    with open(filename,'wt') as f:
        for cls_idx, cls in enumerate(classes):
            dets = all_boxes[cls_idx][im_idx]
            for k in xrange(len(dets)):
                f.write('{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {:.3f}\n'.format(cls.lower(), dets[k][0],  dets[k][1], dets[k][2], dets[k][3],dets[k][4]))

cmd = eval_tool + ' ' \
        + os.path.join(data_root_path, 'training') + ' ' \
        + os.path.join(data_root_path, 'ImageSets', 'val.txt') + ' ' \
        + os.path.dirname(det_file_dir) + ' ' + str(num_images)

print('Running: {}'.format(cmd))
status=subprocess.call(cmd,shell=True)

aps=[]
names=[]

for cls in classes:
    det_file_name=os.path.join(os.path.dirname(det_file_dir), 'stats_{:s}_ap.txt'.format(cls))
    if os.path.exists(det_file_name):
        with open(det_file_name, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3, \
                    'Line number of {} should be 3'.format(det_file_name)
            aps.append(float(lines[0].split('=')[1].strip()))
            aps.append(float(lines[1].split('=')[1].strip()))
            aps.append(float(lines[2].split('=')[1].strip()))
    else:
        aps.extend([0.0,0.0,0.0])

    names.append(cls+'_easy')
    names.append(cls+'_medium')
    names.append(cls+'_hard')

print aps
print names


# And then down here you could detect a lot more images like:
#r = dn.detect(net, meta, "data/eagle.jpg")
#print r
#r = dn.detect(net, meta, "data/giraffe.jpg")
#print r
#r = dn.detect(net, meta, "data/horses.jpg")
#print r
#r = dn.detect(net, meta, "data/person.jpg")
#print r

