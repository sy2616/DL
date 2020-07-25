import tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets,Sequential
import os,time
tf.random.set_seed(220)

cov_layers=[
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same')
]

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255
    y=tf.cast(y,dtype=tf.int32)
    return x,y
(x,y),(x_test,y_test)=datasets.cifar100.load_data()
# print(x.shape,y.shape,x_test.shape,y_test.shape)
y=tf.squeeze(y)
y_test=tf.squeeze(y_test)
print(x.shape,y.shape,x_test.shape,y_test.shape)

bachsz=64
train_db=tf.data.Dataset.from_tensor_slices((x,y))
train_db=train_db.map(preprocess).shuffle(1000).batch(bachsz)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db=test_db.map(preprocess).batch(bachsz)
sample=next(iter(train_db))
print('sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),
      tf.reduce_max(sample[0]))

def main():
    epochs=50
    cov_net=Sequential(cov_layers)
    cov_net.build(input_shape=[None,32,32,3])
    # x=tf.random.normal([4,32,32,3])
    # out=cov_net(x)
    # print(out.shape)
    fc_net=Sequential([
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dense(100,activation=tf.nn.relu)
    ])
    fc_net.build(input_shape=[None,512])
    optimizer=optimizers.Adam(lr=1e-4)
    to_correct=0
    to_num=0

    variables=cov_net.trainable_variables+fc_net.trainable_variables
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out=cov_net(x)
                out=tf.reshape(out,[-1,512])
                logits=fc_net(out)
                y_onehot=tf.one_hot(y,depth=100)
                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))

            if step%100==0:
                print('epoch:',epoch,'step:',step,'loss:',float(loss))
        for x,y in test_db:
            out=cov_net(x)
            out=tf.reshape(out,[-1,512])
            logits=fc_net(out)
            prob=tf.nn.softmax(logits,axis=1)
            pred=tf.argmax(prob,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct=tf.reduce_sum(correct)
            to_num+=x.shape[0]
            to_correct+=int(correct)
        acc=to_correct/to_num
        print('epoch:',epoch,'acc:',acc)



if __name__ == '__main__':
    main()