import tensorflow as tf

'''Read all node in tensorflow graph'''

# Read the graph.
with tf.gfile.FastGFile('model1.pb', 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        print(ts)
    #tf.summary.FileWriter('logs', graph_def)
