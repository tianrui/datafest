import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import math

def tf_eucl_dist(a, b):
    """ Compute the square Euclidean distance between input matrices
    Input:
        a: BxD matrix
        b: KxD matrix
    Output:
        c: BxK matrix

    """
    B, D = a.get_shape()
    K, D = b.get_shape()
    aa = tf.reduce_sum(tf.pow(a,2), 1, keep_dims=True)
    bb = tf.transpose(tf.reduce_sum(tf.pow(b,2), 1, keep_dims=True))
    ab = tf.matmul(a, tf.transpose(b))


    return tf.add(tf.add(aa, bb), tf.mul(ab, -2))

def main1_2():
    # Load the data
    data2D = np.load("data2D.npy")

    # Set constants.
    K = 3
    DATASET_SIZE, DATA_DIM  = data2D.shape
    LEARNINGRATE = 0.01
    ITERATIONS = 500
        
    # Initialize tf graph.
    graph = tf.Graph()
    with graph.as_default():
        # Load data into tf.
        tf_data2D = tf.cast(tf.constant(data2D), tf.float32)
        
        # Initialize mu array.
        tf_mu = tf.Variable(tf.truncated_normal([K, DATA_DIM], dtype=tf.float32, stddev=1.0/np.sqrt(DATA_DIM)))
        
        ed = tf_eucl_dist(tf_data2D, tf_mu)
        
        cluster_assignments = tf.argmin(ed, 1)
        
        loss = tf.reduce_sum(tf.reduce_min(ed, 1))

        optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    # Run session.
    with tf.Session(graph=graph) as session:
        
        losses = np.zeros(ITERATIONS, dtype=np.float32)
        tf.initialize_all_variables().run()

        for i in range(ITERATIONS):
            _, ca, l, m = session.run([optimizer, cluster_assignments, loss, tf_mu])
            losses[i] = l
            
        print ca
        red = [1, 0, 0]
        green = [0, 1, 0]
        blue = [0, 0, 1]
        colours = [red, green, blue]
        colour_list = [colours[ca[i]] for i in range(DATASET_SIZE)]
        
        # Plot data points labelled by the closest mean  
        plt.scatter(data2D[:,0], data2D[:,1], c=colour_list, marker='.')
        # Plot mean
        plt.scatter(m[:,0], m[:,1], marker='h')
        print m
        # TODO: Add plot title, axis labels
        plt.show()

    return 

def main1_3():
    """
    Run the algorithm with K = 1, 2, 3, 4, 5 and for each of these values of K, 
    compute and report the percentage of the data points belonging to each of the K clusters.
    Comment on how many clusters you think is “best” and discuss this value in the context 
    of a 2D scatter plot of the data. Include the 2D scatter plot of data points colored
    by their cluster assignments.
    """
    
    # Load the data
    data2D = np.load("data2D.npy")

    # Set constants.
    DATASET_SIZE, DATA_DIM  = data2D.shape
    LEARNINGRATE = 0.01
    ITERATIONS = 500
    Ks = range(1, 6)
    
    for K in Ks:
        # Initialize tf graph.
        graph = tf.Graph()
        with graph.as_default():
            # Load data into tf.
            tf_data2D = tf.cast(tf.constant(data2D), tf.float32)

            # Initialize mu array.
            tf_mu = tf.Variable(tf.truncated_normal([K, DATA_DIM], dtype=tf.float32, stddev=1.0/np.sqrt(DATA_DIM)))

            ed = tf_eucl_dist(tf_data2D, tf_mu)

            cluster_assignments = tf.argmin(ed, 1)

            loss = tf.reduce_sum(tf.reduce_min(ed, 1))

            optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

        # Run session.
        with tf.Session(graph=graph) as session:

            losses = np.zeros(ITERATIONS, dtype=np.float32)
            tf.initialize_all_variables().run()

            for i in range(ITERATIONS):
                _, ca, l, m = session.run([optimizer, cluster_assignments, loss, tf_mu])
                losses[i] = l

            cluster_bincount = np.bincount(ca)
            cluster_binpercent = [float(x)/DATASET_SIZE for x in cluster_bincount]
        
            red = [1, 0, 0]
            green = [0, 1, 0]
            blue = [0, 0, 1]
            cyan = [0, 1, 1]
            yellow = [1, 1, 0]
            colours = [red, green, blue, cyan, yellow]
            colour_list = [colours[ca[i]] for i in range(DATASET_SIZE)]

            # Plot data points labelled by the closest mean  
            plt.figure()
            plt.scatter(data2D[:,0], data2D[:,1], c=colour_list, marker='.')
            # Plot mean
            plt.scatter(m[:,0], m[:,1], marker='h', s=200)
            plt.show()
            
            # Print results
            print "\n\n    %d cluster(s):" % K
            print "Cluster centers:"
            print m
            print "Percentage of data in each cluster:"
            print cluster_binpercent

    return

def main1_4(data):
    """
    Hold 1/3 of the data out for validation. For each value of K above, cluster the training data
    and then compute and report the loss for the validation data. How many clusters do you
    think is best?
    """
        
    # Load the data
    data2D = np.load(data)
    # Shuffle data
    #np.random.shuffle(data2D)

    # Set constants.
    DATASET_SIZE, DATA_DIM  = data2D.shape
    LEARNINGRATE = 0.01
    ITERATIONS = 2000
    Ks = range(1, 6)
    
    third = DATASET_SIZE / 3
    val_data = data2D[:third]
    train_data = data2D[third:]
    
    for K in Ks:
        # Initialize tf graph.
        graph = tf.Graph()
        with graph.as_default():
            # Training
            # Load data into tf.
            tf_train_data = tf.cast(tf.constant(train_data), tf.float32)

            # Initialize mu array.
            tf_mu = tf.Variable(tf.truncated_normal([K, DATA_DIM], dtype=tf.float32, stddev=1.0/np.sqrt(DATA_DIM)))
            # Euclidean Distance
            ed_train = tf_eucl_dist(tf_train_data, tf_mu)
            # Cluster Assignments
            ca_train = tf.argmin(ed_train, 1)
            # Loss
            loss_train = tf.reduce_sum(tf.reduce_min(ed_train, 1))
            # Optimizer
            optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss_train)
            
            # Validation
            
            # Load validation data.
            tf_val_data = tf.cast(tf.constant(val_data), tf.float32)
            
            # Euclidean Distance
            ed_val = tf_eucl_dist(tf_val_data, tf_mu)
            # Cluster Assignments
            ca_val = tf.argmin(ed_val, 1)
            # Loss
            loss_val = tf.reduce_sum(tf.reduce_min(ed_val, 1))

        # Run session.
        with tf.Session(graph=graph) as session:

            losses = np.zeros(ITERATIONS, dtype=np.float32)
            tf.initialize_all_variables().run()

            for i in range(ITERATIONS):
                _, m, ca, l = session.run([optimizer, tf_mu, ca_val, loss_val])

#           print ca
            cluster_bincount = np.bincount(ca)
            cluster_binpercent = [float(x)/third for x in cluster_bincount]
        
            red = [1, 0, 0]
            green = [0, 1, 0]
            blue = [0, 0, 1]
            cyan = [0, 1, 1]
            yellow = [1, 1, 0]
            colours = [red, green, blue, cyan, yellow]
            colour_list = [colours[ca[i]] for i in range(ca.shape[0])]
    #         print colour_list

            # Plot data points labelled by the closest mean  
            plt.figure()
            plt.scatter(val_data[:,0], val_data[:,1], c=colour_list, marker='.')
            # Plot mean
            plt.scatter(m[:,0], m[:,1], marker='h', s=200)
            plt.show()

            # Print results
            print "\n\n    %d cluster(s):" % K
            print "Cluster centers:"
            print m
            print "Percentage of data in each cluster:"
            print cluster_binpercent
    
    return

def tf_log_pdf_clust(x, mu, sigma, D):
    """ Compute the log probability density function for cluster k for all pair of B data points and K Clusters
        Input:
            x: BxD matrix
            mu: KxD matrix
            sigma: Kx1 matrix
            D: Dimension
        Output:
            c: BxK matrix: log pdf for cluster k
    """
    eucld_dist = tf_eucl_dist(x, mu)
    c = tf.mul(-0.5*D, tf.log(2*math.pi*sigma)) - tf.div(eucld_dist, 2*sigma)
    
    return c

def log_posterior(x, mu, sigma, pi, D):
    """ Compute the probability of the cluster variable z given the data vector x
        Input:
            x: BxD matrix
            mu: KxD matrix
            sigma: 1xK matrix
            pi: 1xK matrix
            D: Dimension 
        Output:
            c: BxK matrix: log P(z|x)
    """
    return tf.log(pi) + tf_log_pdf_clust(x, mu, sigma, D) - utils.reduce_logsumexp(tf_log_pdf_clust(x,mu,sigma, D)+tf.log(pi), keep_dims=True)

def mog_k3():
    # Load the data
    data2D = np.load("data2D.npy")

    # Set constants.
    K = 3
    DATASET_SIZE, DATA_DIM  = data2D.shape
    LEARNINGRATE = 0.01
    ITERATIONS = 750
    
    # Initialize tf graph.
    graph = tf.Graph()
    with graph.as_default():
        # Load data into tf.
        tf_data2D = tf.cast(tf.constant(data2D), tf.float32)
        
        # Initialize mu array.
        tf_mu = tf.Variable(tf.truncated_normal([K, DATA_DIM], dtype=tf.float32, stddev=1.0))
        tf_phi = tf.Variable(tf.truncated_normal([1, K], dtype=tf.float32, mean=1.0, stddev=1.0/np.sqrt(DATA_DIM)))
        tf_sig_sq = tf.exp(tf_phi)
        tf_psi = tf.Variable(tf.truncated_normal([1, K], dtype=tf.float32, mean=1.0,stddev=1.0/np.sqrt(DATA_DIM)))
        tf_pi = tf.exp(utils.logsoftmax(tf_psi))
    
        ed = tf_eucl_dist(tf_data2D, tf_mu)
        loss = -tf.reduce_sum(utils.reduce_logsumexp(tf_log_pdf_clust(tf_data2D,tf_mu,tf_sig_sq, DATA_DIM)+tf.log(tf_pi), reduction_indices=1))
        posterior = tf.exp(log_posterior(tf_data2D, tf_mu, tf_sig_sq, tf_pi, DATA_DIM))
        cluster_hard_assignment = tf.argmax(posterior, 1)
        weight = tf.constant([[0, 0.5, 1.0]]) # TODO: Replace this with linspace as func of K
        cluster_soft_assignment = tf.reduce_sum(tf.mul(weight, posterior), reduction_indices=1)
        optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    # Run session.
    with tf.Session(graph=graph) as session:
        
        losses = np.zeros(ITERATIONS, dtype=np.float32)
        tf.initialize_all_variables().run()

        for i in range(ITERATIONS):
            mu, sig_sq, psi, pi, ca, ca_soft, post = session.run([tf_mu, tf_sig_sq, tf_psi, tf_pi, cluster_hard_assignment, cluster_soft_assignment, posterior])
            _, l, m = session.run([optimizer, loss, tf_mu])
            losses[i] = l
            if i % 100 == 0:
                print "Loss at iteration %d: " % (i), l 
            
        print "Mu:"
        print mu
        print "Sigma:"
        print sig_sq
        print "Pi:"
        print pi
        print "Posterior:"
        print post
        print "Cluster hard assignment:"
        print ca
        red = [1, 0, 0]
        green = [0, 1, 0]
        blue = [0, 0, 1]
        colours = [red, green, blue]
        colour_list = [colours[ca[i]] for i in range(DATASET_SIZE)]
        
        # Plot data points labelled by the closest mean  
        plt.scatter(data2D[:,0], data2D[:,1], c=colour_list, marker='.')
        # Plot mean
        plt.scatter(m[:,0], m[:,1], marker='h')
        plt.show()
        print m
        
        # Plot soft assignment scatterplots
        # TODO: May be redo it so that C = C1*P(z=1|x) + C2*P(z=1|x) + C3*P(z=1|x)
        # Where C1 = Red, C2 = Green, C3 = Blue. Right now using colourmap 'viridis'
        print "Cluster soft assignment:"
        print ca_soft
        plt.figure()
        plt.scatter(data2D[:,0], data2D[:,1], c=ca_soft, cmap='viridis', marker='.')
        plt.scatter(m[:,0], m[:,1], marker='h')
        plt.title("Soft Assignment to Gaussian Cluster")
        # TODO: Add plot title, axis labels
        plt.show()
    
    return

def main2_2_3(data):
	# Hold out 1/3 of the data for validation and for each value of K = 1; 2; 3; 4; 5, train a MoG
	# model. For each K, compute and report the loss function for the validation data and explain
	# which value of K is best. Include a 2D scatter plot of data points colored by their cluster
	# assignments.

    # Load the data
    data2D = np.load(data)

    # Set constants.
    DATASET_SIZE, DATA_DIM  = data2D.shape
    LEARNINGRATE = 0.01
    ITERATIONS = 750
    
    Ks = range(1, 6)
    
    third = DATASET_SIZE / 3
    val_data = data2D[:third]
    train_data = data2D[third:]
    
    for K in Ks:
        # Initialize tf graph.
        graph = tf.Graph()
        with graph.as_default():
            # Training
            # Load data into tf.
            tf_data2D_train = tf.cast(tf.constant(train_data), tf.float32)

            # Initialize mu array.
            tf_mu = tf.Variable(tf.truncated_normal([K, DATA_DIM], dtype=tf.float32, stddev=1.0))
            tf_phi = tf.Variable(tf.truncated_normal([1, K], dtype=tf.float32, mean=1.0, stddev=1.0/np.sqrt(DATA_DIM)))
            tf_sig_sq = tf.exp(tf_phi)
            tf_psi = tf.Variable(tf.truncated_normal([1, K], dtype=tf.float32, mean=1.0,stddev=1.0/np.sqrt(DATA_DIM)))
    #         tf_pi = tf.nn.softmax(tf_psi) # TODO: Use the utils function instead of the tf.nn.softmax
            tf_pi = tf.exp(utils.logsoftmax(tf_psi))

            ed = tf_eucl_dist(tf_data2D_train, tf_mu)
            loss = -tf.reduce_sum(utils.reduce_logsumexp(tf_log_pdf_clust(tf_data2D_train,tf_mu,tf_sig_sq, DATA_DIM)+tf.log(tf_pi), reduction_indices=1))
            
            optimizer = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
            
            # Validation
            # Load data into tf.
            tf_data2D_val = tf.cast(tf.constant(val_data), tf.float32)
            loss_v = -tf.reduce_sum(utils.reduce_logsumexp(tf_log_pdf_clust(tf_data2D_val,tf_mu,tf_sig_sq, DATA_DIM)+tf.log(tf_pi), reduction_indices=1))
            posterior = tf.exp(log_posterior(tf_data2D_val, tf_mu, tf_sig_sq, tf_pi, DATA_DIM))
            cluster_hard_assignment = tf.argmax(posterior, 1)
            weight = tf.cast(tf.constant(np.linspace(0.0, 1.0, K)), tf.float32)
            cluster_soft_assignment = tf.reduce_sum(tf.mul(weight, posterior), reduction_indices=1)
            
        # Run session.
        with tf.Session(graph=graph) as session:

            losses = np.zeros(ITERATIONS, dtype=np.float32)
            tf.initialize_all_variables().run()

            for i in range(ITERATIONS):
                mu, sig_sq, psi, pi, ca, ca_soft, post = session.run([tf_mu, tf_sig_sq, tf_psi, tf_pi, cluster_hard_assignment, cluster_soft_assignment, posterior])
                _, l, l_v, m = session.run([optimizer, loss, loss_v, tf_mu])
                losses[i] = l
                if i % 10 == 0:
                    print "Loss at iteration %d: " % (i), l_v 

            print "Mu:"
            print mu
            print "Sigma:"
            print sig_sq
            print "Pi:"
            print pi
            print "Posterior:"
            print post
            print "Cluster hard assignment:"
            print ca
            
            red = [1, 0, 0]
            green = [0, 1, 0]
            blue = [0, 0, 1]
            cyan = [0, 1, 1]
            yellow = [1, 1, 0]
            colours = [red, green, blue, cyan, yellow]
            
            colour_list = [colours[ca[i]] for i in range(ca.shape[0])]
    #         print colour_list

            # Plot data points labelled by the closest mean  
            plt.figure()
            plt.scatter(val_data[:,0], val_data[:,1], c=colour_list, marker='.')
            # Plot mean
            plt.scatter(m[:,0], m[:,1], marker='h', s=200)
            plt.show()
            print m

            # Plot soft assignment scatterplots
            print "Cluster soft assignment:"
            print ca_soft
            plt.figure()
            plt.scatter(val_data[:,0], val_data[:,1], c=ca_soft, marker='.')
            plt.scatter(m[:,0], m[:,1], marker='h', s=200)
            plt.title("Soft Assignment to Gaussian Cluster")
            plt.show()

    return

def main2_4():
	# TODO: Run both the K-means and the MoG learning algorithms on data100D.npy. Comment on how
	# many clusters you think are within the dataset and compare the learnt results of K-means
	# and MoG.

	data = "data100D.npy"
	main1_4(data)
	main2_2_3(data)
	
	return

if __name__ == '__main__':
	main1_2()
	main1_3()
	main1_4("data2D.npy")
	
	mog_k3()
	main2_2_3()
	main2_4()
