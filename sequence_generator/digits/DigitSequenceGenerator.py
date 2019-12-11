import numpy as np
import pandas as pd
import logging
from skimage.transform import rescale
from sklearn.datasets import fetch_openml

class DigitSequenceGenerator:
    '''A digit sequence generator.

    A generator for creating images of numbers, 1 or more digits, from the MNIST dataset.
    '''
    def __init__(self):
        self.mnist_data = self._load_mnist_data()

    def _load_mnist_data(self):
        '''Loads the MNIST dataset from the openml database.

        Returns
            Dictionary containing samples of digits from 0-9.
        '''
        mnist = fetch_openml('MNIST_784')
        mnist_data = pd.DataFrame(mnist['data'], index=mnist['target']).apply(np.array, axis=1).reset_index().rename(columns={'index': 'target', 0:'data'})
        mnist_data.target = mnist_data.target.astype(int)
        mnist_data = mnist_data.groupby('target')['data'].apply(list).to_dict()

        return mnist_data

    def _crop_spacing(self, img):
        '''Takes an image and crops out all columns that are empty.

        Parameters
            img: 2-d numpy array representing an image

        Returns
            2-d numpy array
        '''
        return img[:, ~np.all(img == 0, axis=0)]

    def _expand_spaces(self, height, width):
        '''Expand an integer representation of spaces into a 2-d array representation of an image.

        Parameters
            height: int
                Expected height of the spaces
            width: int
                Expected width of the spaces

        Returns
            2-d numpy array
        '''
        return np.zeros((height, width))

    def _generate_spaces(self, num_spaces, min_spacing, max_spacing, max_total_spacing):
        '''Generates integer representations of spacing blocks to be used at the edges of the image and between each digit while ensuring that image width and min and max spacing constraints can be met.

        Parameters
            num_spaces: int
                number of space blocks to be generated
            min_spacing: int
                minimum number of spaces to be included in each space block
            max_spacing: int
                maximum number of spaces to be included in each space block
            max_total_spacing: int
                the maximum total of spaces that can be generated while ensuring that there is still enough space allocated for the digits

        Returns
            1-d numpy array
        '''
        spaces = []
        for num_remaining in range(num_spaces-1, -1, -1):
            max_spacing = min(max_spacing, max_total_spacing - num_remaining*min_spacing)
            spaces.append(np.random.randint(low=min_spacing, high=max_spacing+1))
            max_total_spacing -= spaces[-1]

        np.random.shuffle(spaces) # for handling bias in space sizing for tight search spaces

        return np.array(spaces)

    def _check_image_width(self, img, image_width):
        '''Checks if the generated image has the correct width as specified in input.

        Parameters
            img: 2-d numpy array
                numpy representation of an image
            image_width: int
                expected image width of image
        '''
        if img.shape[1] != image_width:
            logging.warning('Incorrect image width')
            logging.warning('Actual width: {}'.format(img.shape[1]))
            logging.warning('Target width: {}'.format(image_width))

    def _check_spacing(self, img, min_spacing, max_spacing):
        '''Checks if the generated image has spaces that meet the min/max spacing requirements as specified in input.

        Parameters
            img: 2-d numpy array
                numpy representation of an image
            min_spacing: int
                minimum size of space block specified as part of input
            max_spacing: int
                maximum size of space block specified as part of input
        '''
        all_consecutive_zeros = (img.sum(axis=0) == 0)
        spaces = np.diff(np.where(np.concatenate(([all_consecutive_zeros[0]],
                             all_consecutive_zeros[:-1] != all_consecutive_zeros[1:],
                             [True])))[0])[::2]

        if (spaces >= min_spacing).mean() != 1 or (spaces <= max_spacing).mean() != 1:
            logging.warning('Incorrect spacing')
            logging.warning('Spacing limits are: {} {}'.format(min_spacing, max_spacing))
            logging.warning('Img spacing is: {}'.format(spaces))

    def _adjust_rounding_error(self, imgs, target_width, spaces, min_spacing, max_spacing):
        '''Adjusts for any rounding errors made in image scaling by scaling the allocated spacing.

        Each digit is scaled to a size that meets image width requirements, taking into consideration spacing constraints. Scaling is done as a proportion which introduces errors where digit sizes are not what is expected. Correct for these by randomly adjusting spacing while ensuring constraints are still met.

        Parameters
            imgs: iterable of 2-d numpy arrays
                image representations of digits
            target_width: int
                target width of image taking into consideration spacing
            spaces: 1-d numpy array
                array containing space representation between digits
            min_spacing: int
                minimum spacing constraint
            max_spacing: int
                maximum spacing constraint

        Returns
            1-d numpy array
        '''
        scaled_width = sum([i.shape[1] for i in imgs])
        rounding_error = scaled_width - target_width

        if rounding_error > 0:
            for _ in range(rounding_error):
#                 print(spaces)
                spaces[np.random.choice(np.where(spaces > min_spacing)[0])] -= 1
        elif rounding_error < 0:
            for _ in range(abs(rounding_error)):
#                 print(spaces)
                spaces[np.random.choice(np.where(spaces < max_spacing)[0])] += 1

        return spaces

    def _generate_digit_imgs(self, digits, target_width):
        '''Generates image representation of digits

        Parameters
            digits: iterable of integers representing digits within number to be generated
            target_width: target width of final image taking into account spacing allocation

        Returns
            list of 2-d numpy arrays
        '''
        digit_imgs = [self.mnist_data[d][np.random.choice(len(self.mnist_data[d]))].reshape(28,28) for d in digits]
        digit_imgs = [self._crop_spacing(d) for d in digit_imgs]
        digit_imgs = [rescale(d, target_width / sum([d.shape[1] for d in digit_imgs])) for d in digit_imgs]

        return digit_imgs

    def _generate_digit_sequence(self, digit_imgs, spaces):
        '''Generates an image representation of the number represented by input digits

        Parameters
            digit_imgs: iterable of 2-d numpy array
                image representation of digit components
            spaces: 1-d numpy array
                integer representation of spaces to be included between digits

        Returns
            2-d numpy array
        '''
        spaces = [self._expand_spaces(digit_imgs[0].shape[0], s) for s in spaces]
        digit_sequence = np.hstack([*sum(zip(spaces, digit_imgs), ()), spaces[-1]])

        return digit_sequence

    def create_digit_sequence(self, number, image_width, min_spacing, max_spacing, min_digit_width=3, debug=False, max_tries=5):
        '''Creates an image representing a digit sequence.

        Parameters
            number: str
                A string representing the number, e.g. '14543'
            image_width: int
                The image width (in pixel)
            min_spacing: int
                The minimum spacing between digits (in pixel)
            max_spacing: int
                The maximum spacing between digits (in pixel)
            max_tries: int
                Maximum number of tries to deal with edge case sampling as a result of constrained parameter space

        Returns
            2-d numpy array
        '''

        def _generate_sequence_components():
            '''Pipeline for generating digit images and space blocks

            Handles a rare case where the maximum or minimum number of spaces are allocated but there is rounding error that requires adjustment. Simply run again with different sampling.
            '''
            digit_imgs, spaces = None, None
            for _ in range(max_tries):
                try:
                    spaces = self._generate_spaces(num_spaces, min_spacing, max_spacing, max_total_spacing)
                    target_width = image_width - sum(spaces)

                    digit_imgs = self._generate_digit_imgs(digits, target_width)
                    spaces = self._adjust_rounding_error(digit_imgs, target_width, spaces, min_spacing, max_spacing)
                    break
                except:
                    pass

            if digit_imgs is None or spaces is None:
                raise ValueError('Parameter space is too constrained; consider decreasing min/max spacing or increase image width.')

            return digit_imgs, spaces

        digits = list(map(int, number))
        num_spaces = len(number) + 1
        max_total_spacing = image_width - len(number)*min_digit_width

        if min_spacing*num_spaces > max_total_spacing:
            raise ValueError('Minimum spacing not enough for specified image width, please decrease min/max spacing or increase image width.')

        digit_imgs, spaces = _generate_sequence_components()
        digit_sequence = self._generate_digit_sequence(digit_imgs, spaces)

        if debug:
            self._check_image_width(digit_sequence, image_width)
            self._check_spacing(digit_sequence, min_spacing, max_spacing)

        return digit_sequence


