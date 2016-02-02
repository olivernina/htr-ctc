# htr-ctc
-This tool is under development. There are some issues with the cost function. However, you can try it if you would like
Send me an email to get the transcriptorium data (hwdata.pkl). olivernina@gmail.com


## Usage

you might need to create a python3 environment first if you don't have it i.e.:

```conda create -n py3k python=3 anaconda```

To run the training of ctc in GPU. first activate the python3 environment with 

```source activate py3k```

you can see the tutorial https://www.continuum.io/content/python-3-support-anaconda
run inside rnn_ctc

```python3 train.py ../theano-ctc/hwdata.pkl```


## References
* Graves, Alex. **Supervised Sequence Labelling with Recurrent Neural Networks.** Chapters 2, 3, 7 and 9.
 * Available at [Springer](http://www.springer.com/engineering/computational+intelligence+and+complexity/book/978-3-642-24796-5)
 * [University Edition](http://link.springer.com/book/10.1007%2F978-3-642-24797-2) via. Springer Link.
 * Free [Preprint](http://www.cs.toronto.edu/~graves/preprint.pdf)

## Credits
* rnn-ctc : https://github.com/rakeshvar/rnn_ctc
* Theano implementation of CTC by [Shawn Tan](https://github.com/shawntan/theano-ctc/)

## Dependencies
* Numpy
* Theano

Can easily port to python2 by adding lines like these where necessary:
``` python
from __future__ import print_function
```
