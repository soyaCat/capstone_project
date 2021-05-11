import tensorflow as tf
import matplotlib.pyplot as plt
W=tf.Variable(tf.random_normal([1]))
cost=tf.square(W-3)
x=[]
for i in range(-1000,1000,1):
    x.append(i/100)
y=[]
for i in range(-1000, 1000, 1):
    i=i/100
    y.append((i-3)*(i-3))
plt.plot(x,y)
plt.show()
train=tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    sess.run(train)
    print(sess.run(W))