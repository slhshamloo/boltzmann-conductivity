Convenience Loader
==================

In our research, we frequently load data structured in a specific way from files.
To be more precise, usually, one parameter is varied over one experiment,
while the other parameters are fixed. So, the files are separated by the value of
the parameters that are fixed in each experiment, and the first column of each file
is the independently varied parameter, while the other columns are the results of the
experiment. This is a very common case in our research, and this class helps facilitate
the repetitive task of loading such data.

.. autoclass:: elecboltz.Loader
   :members:
