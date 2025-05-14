# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--n_route', type=int, default=11)
parser.add_argument('--n_route', type=int, default=11)
parser.add_argument('--n_his', type=int, default=84)
parser.add_argument('--n_pred', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('NYC_node_attr.csv'))
    #W = weight_matrix(pjoin('PMU_node_attr.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = 'NYC_data_11_nodes_10080.csv'
#data_file = 'PMU_P.csv'
n_train, n_val, n_test =25,5,5
# n_train, n_val, n_test =5,1,1
# n_train, n_val, n_test =107,16,16
# n_train, n_val, n_test =136,20,20
# n_train, n_val, n_test =  1266,180,180
PeMS = data_gen(pjoin(data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    #model_train(PeMS, blocks, args)
    # model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
    model_test(PeMS, args.batch_size, n_his, n_pred, args.inf_mode)
