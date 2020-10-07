FCE-public for error detection
=================================

This is a version of the publicly released FCE dataset (Yannakoudakis et al., 2011), modified for experiments on error detection.
Rei & Yannakoudakis (2016) describe the creation of this version, and report the first error detection results using the corpus. 
If you're using this dataset, please reference these two papers.

The original dataset contains 1141 examination scripts for training, 97 for testing, and 6 for outlier experiments.
We randomly separated 80 training scripts as a development set, use the same test set, and do not make use of the outlier scripts.

The texts were first tokenised and sentence split by RASP (Briscoe et al., 2006) and then each token was assigned a binary label. 
Tokens that have been annotated within an error tag are labeled as incorrect (i), otherwise they are labeled as correct (c).
When dealing with missing words or phrases, the error label is assigned to the next token in the sequence.
FCE has american spelling marked as erroneous (error type "SA"), which we ignore in this dataset and regard as correct.

The *tsv* directory contains the files in a tab-separated format. Each line contains one token, followed by a tab and then the error label. Sentences are separated by an empty line.
The *ids* directory contains the ids of the examination scripts.
The *filenames* directory contains the names of the files used for different splits.



References
-----------------------

Compositional Sequence Labeling Models for Error Detection in Learner Writing
Marek Rei and Helen Yannakoudakis
In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)

A New Dataset and Method for Automatically Grading ESOL Texts
Helen Yannakoudakis, Ted Briscoe and Ben Medlock
In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL-2011)

The second release of the RASP system
Ted Briscoe, John Carroll and Rebecca Watson
In Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions
