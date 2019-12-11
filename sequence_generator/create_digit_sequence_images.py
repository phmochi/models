#!/usr/bin/env python

from optparse import OptionParser
from digits import DigitSequenceGenerator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


NAME_LEN = 10

def main():
    parser = OptionParser(usage='usage: DigitSequenceGenerator')
    parser.add_option('-n', '--num_samples', default=10, help='number of images to generate', type='int')
    parser.add_option('-l', '--lower_bound', default=0, help='integer lower bound for numbers to generate', type='int')
    parser.add_option('-u', '--upper_bound', default=999999, help='integer upper bound for numbers to generate', type='int')
    parser.add_option('-w', '--image_width', default=100, help='width of output image (in pixels)', type='int')
    parser.add_option('--min_spacing', default=1, help='minimum spacing to have between digits (in pixels)', type='int')
    parser.add_option('--max_spacing', default=5, help='maximum spacing to have between digits (in pixels)', type='int')
    parser.add_option('--min_digit_width', default=3, help='minimum width (in pixels) to allocate for each digit', type='int')
    parser.add_option('--max_tries', default=5, help='maximum number of times to try generating each image for handling edge case sampling as a result of constrained parameter space', type='int')
    parser.add_option('--debug', help='output warning information (enter any value to set True)', action='store_true')

    options, _ = parser.parse_args()
    options = vars(options)

    if not os.path.exists('images'):
        os.makedirs('images')

    dsg = DigitSequenceGenerator()

    labels = {}
    for i in range(options['num_samples']):
        number = str(np.random.randint(options['lower_bound'], options['upper_bound']))
        digits_seq = dsg.create_digit_sequence(number, options['image_width'], options['min_spacing'], options['max_spacing'], options['min_digit_width'], options['debug'], options['max_tries'])

        plt.imshow(digits_seq)
        filename = str(i).zfill(NAME_LEN)
        plt.savefig('images/{}.png'.format(filename))

        labels[filename] = number

    pd.DataFrame.from_dict(labels, orient='index', columns=['label']).to_csv('labels.csv', index_label='filename')

if __name__ == '__main__':
    main()
