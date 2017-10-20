import dataset
import tensorflow as tf
import datetime

# network hyperparameters
num_lstm_units = 64
num_classes = 2
max_iterations = 100000
tensorboard_update_interval = 50


def main():

    dataset.preprocess()

    # prepare the data generator

    training_positive = open(dataset.positive_training_file, 'r', encoding='utf-8')
    training_negative = open(dataset.negative_training_file, 'r', encoding='utf-8')

    training_data_generator = dataset.data_batch_generator(training_positive, training_negative, 250)

    # define lstm network graph

    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')  # auto determine batch size
    data = tf.placeholder(tf.float32, [None, None, dataset.word_vector_size], name='data')  # auto sequence length

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_lstm_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)

    value, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([num_lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    last = value[:, -1, :]  # get the last hidden state in the model

    prediction = (tf.add(tf.matmul(last, weight), bias, name='prediction'))

    # visualise accuracy

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # define loss function and optimizer

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # run training

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # define Tensorboard hooks

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(max_iterations):

        # get next data batch

        try:
            crt_batch, crt_batch_labels = next(training_data_generator)
        except StopIteration:
            print("end of dataset")

            save_path = saver.save(sess, "models2/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

            break

        sess.run(optimizer, {data: crt_batch, labels: crt_batch_labels})

        # write summary to Tensorboard logs

        if i % tensorboard_update_interval is 0:
            summary = sess.run(merged, {data: crt_batch, labels: crt_batch_labels})
            writer.add_summary(summary, i)

    writer.close()

    training_positive.close()
    training_negative.close()

if __name__ == '__main__':
    main()
