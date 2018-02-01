import tensorflow as tf
import numpy as np
x=tf.placeholder(shape=(4,2),dtype=tf.int32)
y=tf.Variable(tf.random_normal(shape=(4,5,3)))
#z=tf.einsum("ijk,ikl->ijl",x,y)
#z=tf.matmul(x,y)
#z=tf.transpose(x,perm=[0,2,1])

z=tf.one_hot(x,6,1,0,1)
z=tf.reshape(z,(4,2,6))


with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    ss=np.random.randint(0,5,size=(4,2))
    z1=sess.run(z,feed_dict={x:ss})
    print(z1)
    #print(z1)
