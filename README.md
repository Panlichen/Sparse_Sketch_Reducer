# Sparse_Sketch_Reducer

We implement the sparse sketch reducer (S2 Reducer) on pytorch 1.9. 
Folder LSTM/ contains the LSTM training code, which is a custom training code based on 'PowerSGD'.
Folder GNMT/ contains the GNMT-8/16 training code, which is a custom training code based on the benchmark of MLPref.
These two models yield sparse gradient from embedding matrix.
./codings contains the main code of sparse-sketch.  