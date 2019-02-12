"""
Title:      Helicopter Wing Detector and Inverse Kinematics
Author:     James Beattie
Created:    22/02/2018
"""

# Imports
########################################################################################################

import numpy as np;
import imageio;                                 # reading in mp4 data
import matplotlib.pyplot as plt;
import skimage;                                 # import image data
from skimage.filters import try_all_threshold;  # thresholding the image
from skimage import filters, measure, data, io, segmentation, color;
from skimage.future import graph;
from skimage.morphology import square;
import matplotlib.patches as patches;
import matplotlib.patches as mpatches;
import argparse;
import colorsys;


# Command Line Arguements
########################################################################################################

ap 			= argparse.ArgumentParser(description = 'Input arguments');
ap.add_argument('-video','--video', required=False, help = 'specfiy the mp4 file');
ap.add_argument('-frame', '--frame',required=False, help = 'the total number of frames to run', type=int);
args 		= vars(ap.parse_args());

########################################################################################################


# Functions Declarations
########################################################################################################
def read_video(vid_file=args['video']):

    vid     = imageio.get_reader(vid_file,  'ffmpeg');
    return vid

def label_creator(image,colour):
    all_labels      = measure.label(image)

    for region in measure.regionprops(all_labels):

        # skip small images
        if region.area < 50:
            continue

        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=colour, linewidth=2)
        ax3.add_patch(rect)

def image_open_func(image):
    """ This function opens the pixel spaces until there are two regions"""
    openfactor      = 10;
    image_open      = skimage.morphology.opening(image,square(openfactor));
    all_labels      = measure.label(image_open);

    while len(measure.regionprops(all_labels)) == 1:
        openfactor      += 10;
        image_open      = skimage.morphology.opening(image,square(openfactor));
        all_labels      = measure.label(image_open);

    return image_open


########################################################################################################

# Working Script
########################################################################################################

# Declate the number of frames
nums            = args['frame'];

# Loop through each frame
for num in xrange(100,nums):

    vid         = read_video();
    image_      = vid.get_data(num);
    image_size  = image_[:,:,0].shape

    # Segment Image
    labels1 = segmentation.slic(image_, compactness=10, n_segments=400);
    out1    = color.label2rgb(labels1, image_, kind='avg');
    g       = graph.rag_mean_color(image_, labels1);
    labels2 = graph.cut_threshold(labels1, g, 30);
    out2    = color.label2rgb(labels2, image_, kind='avg');


    # Create a binary threshold using triangle thresholding.
    BinaryThreshold             = labels2>filters.threshold_triangle(labels2);
    image_centering             = np.zeros(BinaryThreshold.shape);
    image_centering[200:1000,:] = BinaryThreshold[200:1000,:];
    image_open                  = image_open_func(image_centering);
    out3                        = color.label2rgb(image_open, image_, kind='avg');


    ########################
    # Visualisation
    ########################

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8), sharey=True, dpi=150)

    ax1.imshow(out2);
    ax1.axis('off');
    ax1.set_title('Region Adjaceny Graph Seg: {} classes'.format(labels2.max()+1));

    ax2.imshow(out3);
    ax2.axis('off');
    ax2.set_title('Binary Triangle Threshold');

    ax3.imshow(image_);
    ax3.axis('off');
    ax3.set_title('Original');

    # Label Regions
    label_creator(image_centering,'red');
    label_creator(image_open,'green');

    f.tight_layout();
    f.savefig('helic_{}_RAGCUT{}'.format(args['video'],num) + '.png');
    plt.close(f);
