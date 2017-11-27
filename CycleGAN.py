"""Code for training CycleGAN."""
from __future__ import print_function
from datetime import datetime
import json
import numpy as np
import os
import random
import time
import tensorflow as tf

import data_loader, losses, model
from sklearn.neighbors import NearestNeighbors

slim = tf.contrib.slim


class CycleGAN:
    """The CycleGAN module."""

    def __init__(self, do_train, do_test, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, network_version,
                 train_dataset_name, test_dataset_name, checkpoint_dir, skip):
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._do_train=do_train
        self._do_test=do_test
        self._pool_size = pool_size
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        # self._output_dir = os.path.join(output_root_dir, current_time)
        self._output_dir = output_root_dir
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._train_dataset_name = train_dataset_name
        self._test_dataset_name = test_dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._skip = skip

        self.fake_word_A = np.zeros(
            (self._pool_size, model.BATCH_SIZE, model.WORD_EMBED_DIM)
        )
        self.fake_word_B = np.zeros(
            (self._pool_size, model.BATCH_SIZE, model.WORD_EMBED_DIM)
        )

    def model_setup(self):
        """
        This function sets up the model to train.

        self.input_A/self.input_B -> Set of training word.
        self.fake_A/self.fake_B -> Generated word by corresponding generator
        of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> word generated after feeding
        self.fake_A/self.fake_B to corresponding generator.
        This is use to calculate cyclic loss
        """
        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                model.WORD_EMBED_DIM
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model.WORD_EMBED_DIM
            ], name="input_B")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.WORD_EMBED_DIM
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.WORD_EMBED_DIM
            ], name="fake_pool_B")

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'word_a': self.input_a,
            'word_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, network=self._network_version, skip=self._skip)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_word_a = outputs['fake_word_a']
        self.fake_word_b = outputs['fake_word_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_word_a = outputs['cycle_word_a']
        self.cycle_word_b = outputs['cycle_word_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        """
        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_word=self.input_a, generated_word=self.cycle_word_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_word=self.input_b, generated_word=self.cycle_word_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        self.g_loss_A = \
            cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        self.g_loss_B = \
            cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        self.d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        self.d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", self.g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", self.g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", self.d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", self.d_loss_B)

    def fake_word_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated word to corresponding
        pool of word.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        print 
        if num_fakes < self._pool_size:
            fake_pool[num_fakes,:,:] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id,:,:]
                fake_pool[random_id,:,:] = fake
                return temp
            else:
                return fake

    def run(self):
        BATCH_SIZE=model.BATCH_SIZE
        """Training Function."""

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        # saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)

            def do_test():
                my_data_loader = data_loader.DataLoaderDisk_bi(self._test_dataset_name,BATCH_SIZE,True)
                max_word = my_data_loader.num
                its = max_word//BATCH_SIZE+1

                reslist= np.zeros([its*BATCH_SIZE,6,model.WORD_EMBED_DIM])
                for i in range(its):
                    input_a,input_b=my_data_loader.next_batch()
                    fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                        self.fake_word_a,
                        self.fake_word_b,
                        self.cycle_word_a,
                        self.cycle_word_b
                    ], feed_dict={
                        self.input_a: input_a,
                        self.input_b: input_b
                    })

                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),0,:]=input_a
                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),1,:]=input_b
                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),2,:]=fake_A_temp
                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),3,:]=fake_B_temp
                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),4,:]=cyc_A_temp
                    reslist[BATCH_SIZE*i:BATCH_SIZE*(i+1),5,:]=cyc_B_temp
                    
                reslist=reslist[:max_word,:,:]

                print ('Test accruacy')
                print ('top1',evaluation(reslist[:,3],reslist[:,0],n_neighbors=1))
                print ('top5',evaluation(reslist[:,2],reslist[:,1],n_neighbors=5))

            if self._do_train:
                # Load Dataset from the dataset folder
                my_data_loader=data_loader.DataLoaderDisk_bi(self._train_dataset_name,BATCH_SIZE,True)
                max_word = my_data_loader.num
                its = max_word//BATCH_SIZE+1

                # Training Loop
                for epoch in range(sess.run(self.global_step), self._max_step):
                    print ('epoch:',epoch)
                    cur=time.time()
                    # saver.save(sess, os.path.join(self._output_dir, "cyclegan"), global_step=epoch)

                    # Dealing with the learning rate as per the epoch number
                    if epoch < 100:
                        curr_lr = self._base_lr
                    else:
                        curr_lr = self._base_lr - \
                            self._base_lr * (epoch - 100) / 100

                    # self.save_word(sess, epoch)

                    for i in range(0, its):
                        if i%(its//10)==0:
                            print("Processing batch {}/{}".format(i, its))

                        input_a,input_b=my_data_loader.next_batch()
                        # Optimizing the G_A network
                        _, fake_B_temp, summary_str,loss = sess.run(
                            [self.g_A_trainer,
                             self.fake_word_b,
                             self.g_A_loss_summ,
                             self.g_loss_A],
                            feed_dict={
                                self.input_a:
                                    input_a,
                                self.input_b:
                                    input_b,
                                self.learning_rate: curr_lr
                            }
                        )
                        writer.add_summary(summary_str, epoch * max_word + i)
                        print(loss)

                        fake_B_temp1 = self.fake_word_pool(
                            self.num_fake_inputs, fake_B_temp, self.fake_word_B)

                        # Optimizing the D_B network
                        _, summary_str,loss= sess.run(
                            [self.d_B_trainer, 
                             self.d_B_loss_summ,
                             self.d_loss_B],
                            feed_dict={
                                self.input_a:
                                    input_a,
                                self.input_b:
                                    input_b,
                                self.learning_rate: curr_lr,
                                self.fake_pool_B: fake_B_temp1
                            }
                        )
                        writer.add_summary(summary_str, epoch * max_word + i)
                        print(loss)

                        # Optimizing the G_B network
                        _, fake_A_temp, summary_str,loss  = sess.run(
                            [self.g_B_trainer,
                             self.fake_word_a,
                             self.g_B_loss_summ,
                             self.g_loss_B],
                            feed_dict={
                                self.input_a:
                                    input_a,
                                self.input_b:
                                    input_b,
                                self.learning_rate: curr_lr
                            }
                        )
                        writer.add_summary(summary_str, epoch * max_word + i)
                        print(loss)

                        fake_A_temp1 = self.fake_word_pool(
                            self.num_fake_inputs, fake_A_temp, self.fake_word_A)

                        # Optimizing the D_A network
                        _, summary_str,loss = sess.run(
                            [self.d_A_trainer, 
                             self.d_A_loss_summ,
                             self.d_loss_A],
                            feed_dict={
                                self.input_a:
                                    input_a,
                                self.input_b:
                                    input_b,
                                self.learning_rate: curr_lr,
                                self.fake_pool_A: fake_A_temp1
                            }
                        )
                        writer.add_summary(summary_str, epoch * max_word + i)
                        print(loss)

                        writer.flush()
                        self.num_fake_inputs += 1


                    sess.run(tf.assign(self.global_step, epoch + 1))
                    print(time.time()-cur,'s for 1 epoch')
                    
                    if self._do_test:
                        do_test()

            if self._do_test:
                do_test()
            
            # coord.request_stop()
            # coord.join(threads)
            writer.add_graph(sess.graph)

def evaluation(predict,real,n_neighbors=1):
    predict=np.array(predict)
    real=np.array(real)
    model=NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(real)
    output=model.kneighbors(predict, return_distance=False) 
    realmap=np.arange(0,predict.shape[0],1)
    total_acc=0.
    for i in range(predict.shape[0]):
        if realmap[i] in output[i]:
            total_acc+=1.
    return total_acc/predict.shape[0]
