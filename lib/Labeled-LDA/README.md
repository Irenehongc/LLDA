# Labeled-LDA
The C implementation of Labeled-LDA with collapsed gibbs sampling estimation. 

Original file:15.11.30 HIGASHI Koichi https://github.com/khigashi1987/Labeled-LDA

For python library:16.1.4 Shinya SUZUKI

This program makes llda.so to use in python with ctypes.

## Usage

* git clone, and make

* call llda.so by ctypes from python as below

```lang:python
from ctypes import *
llda = CDLL("./llda.so")
llda.calculate.argtypes = [c_int, c_double, c_double, c_char_p, c_char_p, c_char_p]
llda.calculate(30, 0.1, 0.1, "corpus_word.txt", "corpus_label.txt", "output")
```
