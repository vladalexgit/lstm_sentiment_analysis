import numpy as np
import spacy
import tensorflow as tf


def main():

    # load the computational graph from disk

    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph('models2/pretrained_lstm.ckpt-853.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models2'))

    graph = tf.get_default_graph()

    data = graph.get_tensor_by_name('data:0')
    prediction = graph.get_tensor_by_name('prediction:0')

    # collect user input

    my_data = input("enter a test review: ")

    # convert input to word vectors

    nlp = spacy.load('en')
    word_vectors = [t.vector for t in nlp(my_data) if t.has_vector]
    net_input = np.zeros([1, len(word_vectors), 300])

    for i, vec in enumerate(word_vectors):
        net_input[0][i] = vec

    # attempt classification

    result = sess.run(prediction, {data: net_input})

    if result[0][0] > result[0][1]:
        print("PREDICTION: POSITIVE REVIEW")
    else:
        print("PREDICTION: NEGATIVE REVIEW")

if __name__ == '__main__':
    main()
