import argparse
import pickle
from print_utils import slab_print
from alphabets import hindu_alphabet, ascii_alphabet
# import ocropy.ocrolib as ocrolib
# import ocropy.ocrolib.native

import numpy as np
# from ocropy.ocrolib import lineest
# from skimage import transform as tf
# from pylab import *

################################ Parse Arguments ###########################
from scribe import Scribe


class Formatter(argparse.RawDescriptionHelpFormatter,
                argparse.ArgumentDefaultsHelpFormatter):
    pass

desc = '''Generate Sequence Data.
    Will output a pkl file, with the sequence of variable length matrices x_i
    and their corresponding variable length labels y_i. The pkl file will also
    contain the character set corresponding to the labels.
    Examples:
         python3 {0} data -ly 5 -a hindu # For random five-digit numbers
         python3 {0} data -lx 60
         python3 {0} data -lx 30 -fixed -noise .1
         python3 {0} data -ly 1 -n 8400
         python3 {0} data -ly 5
         python3 {0} data -hbuf 8 -vbuf 8'''.format(
    __file__)

prsr = argparse.ArgumentParser(description=desc, formatter_class=Formatter)

prsr.add_argument('-a', action='store', dest='alphabet',
                  default='ascii',
                  help='The alphabel to be used: hindu, ascii, etc.')
prsr.add_argument('-noise', action='store', dest='noise', type=float,
                  default=.05, help='Nosie')
prsr.add_argument('-n', action='store', dest='nsamples', type=int,
                  default=1000, help='Number of samples')

prsr.add_argument('-lx', action='store', dest='avg_seq_len', type=int,
                  default='30',
                  help='Average length of each image.')
prsr.add_argument('-fixed', dest='varying_len', action='store_false',
                  help='Set the length of each image to be variable.')

prsr.add_argument('-ly', action='store', dest='nchars', type=int,
                  default=0,
                  help='Fixed length of each label sequence. Overrides lx and varx')

prsr.add_argument('-vbuf', action='store', dest='vbuffer', type=int,
                  default=3, help='Vertical buffer')
prsr.add_argument('-hbuf', action='store', dest='hbuffer', type=int,
                  default=3, help='Horizontal buffer')

prsr.add_argument('output_name', action='store',
                  help='Output will be stored to <output_name>.pkl')

prsr.set_defaults(varying_len=True)
args = prsr.parse_args()


###########################################################################

out_file_name = args.output_name
out_file_name += '.pkl' if not out_file_name.endswith('.pkl') else ''

if args.alphabet == "ascii":
    alphabet = ascii_alphabet
else:
    alphabet = hindu_alphabet

print(alphabet)
scribe = Scribe(alphabet=alphabet,
                noise=args.noise,
                vbuffer=args.vbuffer,
                hbuffer=args.hbuffer,
                avg_seq_len=args.avg_seq_len,
                varying_len=args.varying_len,
                nchars=args.nchars)



# inputs = ocrolib.glob_all(['/media/onina/SSD/fhtw2016/contestHTRtS/BenthamData/1stBatch/Images/Lines/*.png'])
# training_examples = []
# training_examples2 = [ word.strip() for word in open('dictionary.txt') ]
# charset = sorted(list(set(list(ocrolib.lstm.ascii_labels) + list(ocrolib.chars.default))))
# charset = [""," ","~",]+[c for c in charset if c not in [" ","~"]]
# print("# charset size",len(charset))
#
# if len(charset)<200:
#     print( "["+"".join(charset)+"]")
# else:
#     s = "".join(charset)
#     print("["+s[:20],"...",s[-20:]+"]")
#
# codec = ocrolib.lstm.Codec().init(charset)
# network = ocrolib.lstm.SeqRecognizer(48,512,
#     codec=codec,
#     normalize=ocrolib.lstm.normalize_nfkc)
#
# lnorm = lineest.CenterNormalizer()
# lnorm.setHeight(48)
# network.lnorm = lnorm
import os
if os.path.exists('hwdata.npy'): #False:
    training_examples = np.load('hwdata.npy')

# else:
#     for fname in inputs[2:]:
#         line = ocrolib.read_image_gray(fname)
#         import matplotlib.image as mpimg
#         mpimg.imsave('results/orig_'+fname.split('/')[-1],line,cmap=cm.gray)
#
#         afine_tf = tf.AffineTransform(shear=0.5)
#         line = tf.warp(line,afine_tf)
#         mpimg.imsave('results/desl_'+fname.split('/')[-1],line,cmap=cm.gray)
#
#         network.lnorm.measure(amax(line)-line)
#         mpimg.imsave('results/cont_'+fname.split('/')[-1],(amax(line)-line),cmap=cm.gray)
#
#         print(fname)
#         print(line.shape)
#         try:
#             line = network.lnorm.normalize(line,cval=amax(line))
#         except ValueError:
#             print("Normalization error...skipping sample")
#             continue
#         print(line.shape)
#         if line.size<10 or amax(line)==amin(line):
#             print("EMPTY-INPUT")
#             continue
#
#         line = line * 1.0/amax(line)
#         line = amax(line)-line
#         line = line.T
#
#         base,_ = ocrolib.allsplitext(fname)
#         base = base.replace('Images/Lines','Transcriptions')
#         transcript = ocrolib.read_text(base+".txt")
#         import unicodedata
#         transcript = unicodedata.normalize('NFKD',transcript).encode('ascii','ignore')
#         sample = (line,transcript)
#         training_examples.append(sample)
#     np.save('hwdata',training_examples)



xs = []
ys = []
for i in range(args.nsamples):
    x, y = scribe.get_sample()
    xs.append(x)
    ys.append(y)
    print(y, "".join(alphabet.chars[i] for i in y))
    slab_print(x)

print('Output: {}\n'
      'Char set : {}\n'.format(out_file_name, alphabet.chars))
for var,val in vars(args).items():
    print("{:12}: {}".format(var, val))

with open(out_file_name, 'wb') as f:
    pickle.dump({'x': xs, 'y': ys, 'chars': alphabet.chars}, f, -1)