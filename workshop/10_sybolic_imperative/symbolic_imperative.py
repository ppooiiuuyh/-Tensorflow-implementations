import tensorflow as tf

if __name__ == "__main__" :
#=================================
# 2. Symbollic
#=================================
    #variable
    Var = tf.Variable(3)
    print("tf.Variable(3): ",Var)

    #init graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #run graph
    print("output: ", sess.run(Var))





#================================
# 1. imperative
#================================
    #variable
    var = 3
    print(var)
    var = 3 * var
    print( var)

    #variable without allocation
    var2
    print(var2)


