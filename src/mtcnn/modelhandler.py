import os
import sys

from tensorflow.python.framework import graph_util
import os

import tensorflow as tf
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from tools.tools import detect_face_mod

class MTCNNHandler:
    PATH_TO_WEIGHTS= './weights/all_in_one/'

    path_onet = 'weights/frozen/onet.pb'
    path_pnet = 'weights/frozen/pnet.pb'
    path_rnet = 'weights/frozen/rnet.pb'

    THIS_FOLDER= os.path.dirname(__file__)

    def __init__(self):

        self._path_to_weights = self.PATH_TO_WEIGHTS


        self._path_onet = os.path.join(self.THIS_FOLDER, self.path_onet )
        self._path_pnet = os.path.join(self.THIS_FOLDER, self.path_pnet )
        self._path_rnet = os.path.join(self.THIS_FOLDER, self.path_rnet )

        self._detection_graph_onet = None
        self._detection_graph_pnet = None
        self._detection_graph_rnet = None

        self._session_onet = None
        self._session_pnet = None
        self._session_rnet = None

        self._onet_output = None
        self._pnet_output = None
        self._rnet_output = None


        self._threshold = [0.8, 0.8, 0.8]   # Three thresholds for pnet, rnet, onet, respectively.
        self._factor = 0.7                  # The scale stride of orginal image
        self._minsize = 20                  # The minimum size of face to detec

    def setup(self):
        print('SETTING UP')
        self.load_model(self._path_onet, self._path_pnet, self._path_rnet)
        self.detect(np.ones((300, 300, 3), dtype=np.uint8))

    def load_model(self, path_onet, path_pnet, path_rnet):
        """
            Setup the model, if model does not exist in path, this will be downloaded
        """

        # ONet
        with tf.gfile.GFile(path_onet, 'rb') as fid:
            graph_def_onet = tf.GraphDef()
            graph_def_onet.ParseFromString(fid.read())
        
        with tf.Graph().as_default() as graph_onet:
            tf.import_graph_def(graph_def_onet, name='')

        # Pnet
        with tf.gfile.GFile(path_pnet, 'rb') as fid:
            graph_def_pnet = tf.GraphDef()
            graph_def_pnet.ParseFromString(fid.read())

        with tf.Graph().as_default() as graph_pnet:
            tf.import_graph_def(graph_def_pnet, name='')

        # RNET
        with tf.gfile.GFile(path_rnet, 'rb') as fid:
            graph_def_rnet = tf.GraphDef()
            graph_def_rnet.ParseFromString(fid.read())

        with tf.Graph().as_default() as graph_rnet:
            tf.import_graph_def(graph_def_rnet, name='')



        self._detection_graph_onet = graph_onet
        self._detection_graph_pnet = graph_pnet
        self._detection_graph_rnet = graph_rnet

        self._session_onet = tf.Session(graph=self._detection_graph_onet)
        self._session_pnet = tf.Session(graph=self._detection_graph_pnet)
        self._session_rnet = tf.Session(graph=self._detection_graph_rnet)

        for i in self._session_onet.get_operations():
            print(i.name)

        # Definite input and output Tensors for detection_graph

        #self._onet_output = self._detection_graph_onet.get_tensor_by_name(('softmax/Reshape_1:0','pnet/conv4-2/BiasAdd:0'))

        #self._pnet_output = self._detection_graph_pnet.get_tensor_by_name(('softmax_1/softmax:0', 'net/conv5-2/rnet/conv5-2:0'))

        #self._rnet_output = self._detection_graph_rnet.get_tensor_by_name(('softmax_2/softmax:0', 'onet/conv6-2/onet/conv6-2:0', 'onet/conv6-3/onet/conv6-3:0'))

        #self._onet_output = ('softmax/Reshape_1:0','pnet/conv4-2/BiasAdd:0')

        #self._pnet_output = ('softmax_1/softmax:0', 'net/conv5-2/rnet/conv5-2:0')

        #self._rnet_output = ('softmax_2/softmax:0', 'onet/conv6-2/onet/conv6-2:0', 'onet/conv6-3/onet/conv6-3:0')
        
        self._onet_output = self._detection_graph_onet.get_tensor_by_name('softmax/Reshape_1:0')

        self._pnet_output = self._detection_graph_pnet.get_tensor_by_name('softmax_1/softmax:0')

        self._rnet_output = self._detection_graph_rnet.get_tensor_by_name('softmax_2/softmax:0')


    def detect(self, image):
        """
            Detect objects in the image
            :image = Image array, numpy array

            :return: list of the detection outputs.
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.

        def onet_fun(img, output_names_onet, sess_onet ): return sess_onet.run(
            output_names_onet, feed_dict={'Placeholder:0': img})

        def pnet_fun(img, output_names_pnet, sess_pnet ): return sess_pnet.run(
            output_names_pnet, feed_dict={'Placeholder:1': img})

        def rnet_fun(img, output_names_rnet, sess_rnet): return sess_rnet.run(
            output_names_rnet, feed_dict={'Placeholder:2': img})


        
        rectangles, points = detect_face_mod(image_np_expanded, self._minsize,
                                    pnet_fun, self._pnet_output, self._session_pnet,
                                    rnet_fun, self._rnet_output, self._session_rnet,
                                    onet_fun, self._onet_output, self._session_onet,
                                    self._threshold, self._factor)
        

        return rectangles, points


class FreezeModel:
    def __init__(self):
        pass

    @staticmethod
    def freeze_graph(input_checkpoint, output_graph):
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)
        
        #output_node_names = ["cls_prob", "bbox_pred", "landmark_pred"]
        output_node_names = ('softmax/Reshape_1:0','pnet/conv4-2/BiasAdd:0')
        output_graph_def = graph_util.convert_variables_to_constants( sess=sess,
                                                    input_graph_def=sess.graph_def,
                                                    output_node_names=output_node_names) # multi output)
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))

        for op in tf.get_default_graph.get_operations():
            print(op.name, '----->', op.values())

if __name__ == '__main__':

    input_checkpoint = 'weights/all_in_one'

    output_folder = "weights/all_in_one/pb"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    out_pb_path = os.path.join(output_folder, 'frozen.pb')

    freezer = FreezeModel()
    input_checkpoint_file = os.path.join(input_checkpoint, 'mtcnn-3000000')
    freezer.freeze_graph(input_checkpoint_file, out_pb_path)