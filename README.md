# htr-ctc
-This tool is not longer supported and has been discontinued.
Please see our new library in Pytorch which we use on the ICFHR 2018 READ dataset competition where we ranked 2nd place. 
https://github.com/olivernina/nephi


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

