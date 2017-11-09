import tensorflow as tf


def load_data(dataset_name,do_shuffle=True):
    """
    :param dataset_name: The name of the dataset.
    :param word_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """

    word_i=tf.convert_to_tensor(np.load(dataset_name[0]), dtype=tf.float32)
    word_j=tf.convert_to_tensor(np.load(dataset_name[1]), dtype=tf.float32)

    inputs = {
        'word_i': word_i,
        'word_j': word_j
    }

    #To Do normalize

    # Batch
    if do_shuffle is True:
        inputs['words_i'], inputs['words_j'] = tf.train.shuffle_batch(
            [inputs['word_i'], inputs['word_j']], 1, 5000, 100)
    else:
        inputs['words_i'], inputs['words_j'] = tf.train.batch(
            [inputs['word_i'], inputs['word_j']], 1)

    return inputs
