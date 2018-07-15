from k_means_solved import KMeans, KMeansPlusPlus
import numpy as np
import argparse
from matplotlib import image as mpimg
import matplotlib.pyplot as plt


class AlgorithmSelectionAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, eval(value))


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path of img')
    parser.add_argument('-k', help='number of clusters', type=int)
    parser.add_argument('--algorithm', choices=['KMeans', 'KMeansPlusPlus'],
                        action=AlgorithmSelectionAction)
    parser.add_argument('--name', help='type your name')
    return parser.parse_args()

def postprocess(labels, means):
    return np.array([means[int(x)].astype(np.uint8) for x in labels])

def main(args):
    img = mpimg.imread(args.path).astype(float)
    shp = img.shape
    kmeans = args.algorithm(args.k)
    if len(shp) == 2:
        img = img.reshape(list(img.shape) + [1])
    kmeans.fit(img)
    plt.axis('off')
    global new_img
    new_img = postprocess(*kmeans.predict(img))
    new_img = new_img.reshape(shp)
    plt.imshow(new_img, "gray" if len(shp)==2 else None)
    plt.savefig('{}_{}'.format(args.path, args.name).replace('.',''), bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
