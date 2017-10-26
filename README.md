# lstm_sentiment_analysis
Sentiment analysis on movie reviews using LSTM neural networks

# Task

Given a movie review classify it as either positive or negative, after training the agent on an adnotated dataset.

# Dataset

For this task i have used the ["Large Movie Review Dataset" provided by Stanford University](http://ai.stanford.edu/~amaas/data/sentiment/)

You can read their paper on this subject [here](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) :

```
   @InProceedings{maas-EtAl:2011:ACL-HLT2011,
      author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
      title     = {Learning Word Vectors for Sentiment Analysis},
      booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
      month     = {June},
      year      = {2011},
      address   = {Portland, Oregon, USA},
      publisher = {Association for Computational Linguistics},
      pages     = {142--150},
      url       = {http://www.aclweb.org/anthology/P11-1015}
    }
```

# Required packages

   * numpy
   * tensorflow
   * spacy - all word embedding vectors were loaded using spacy

# Usage

If you have not yet downloaded the dataset mentioned above, the script will attempt to download it for you on the first run.

First you should run `train_LSTM.py`. This script will preprocess the dataset and save it in a separate folder. Afterwards it will begin training the LSTM neural network. To visualize the progress in Tensorboard you can run the following command ```tensorboard --logdir=logs```

After training the network you can run the script `test_LSTM.py` to see how well the network fits the test set.

The most exciting script by far is `own_input_test_LSTM.py` where you can enter a review as a string and see how the network classifies it.

   
