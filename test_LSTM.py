import dataset
import tensorflow as tf


num_test_iterations = 20


def main():

    # prepare test data batch generator

    test_positive = open(dataset.positive_test_file, 'r', encoding='utf-8')
    test_negative = open(dataset.negative_test_file, 'r', encoding='utf-8')

    test_data_generator = dataset.data_batch_generator(test_positive, test_negative, 250)

    # load the computational graph from disk

    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph('models2/pretrained_lstm.ckpt-853.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models2'))

    graph = tf.get_default_graph()

    data = graph.get_tensor_by_name('data:0')
    labels = graph.get_tensor_by_name('labels:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')

    for i in range(num_test_iterations):

        try:
            crt_batch, crt_batch_labels = next(test_data_generator)
        except StopIteration:
            print("end of dataset")
            break

        result = sess.run(accuracy, {data: crt_batch, labels: crt_batch_labels})

        print("Accuracy ", i, ":", result)

    test_positive.close()
    test_negative.close()

if __name__ == '__main__':
    main()
