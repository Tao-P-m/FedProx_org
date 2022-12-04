import time
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.DC_DS import DC_NORIS

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {}/{} workers ---'.format(self.M, len(self.clients)))
        self.M = len(self.clients) # dirty fix 26/09/2022
        if self.set==1:
            self.K = np.ones(self.M, dtype=int)*int(30000.0/self.M)
            print("sum of K (set==1)", sum(self.K))
            self.dx2 = np.random.rand(int(self.M - np.round(self.M/2)))*self.range-self.range #[100,100+range]
        else:
            self.K = np.random.randint(1000, high=2001, size=(int(self.M )))
            lessuser_size = int(self.M /2) # bisection
            self.K2 = np.random.randint(100, high=201, size=(lessuser_size))
            self.lessuser = np.random.choice(self.M, size=lessuser_size, replace=False)
            self.K[self.lessuser] = self.K2
            print("sum of K (set==2)", sum(self.K), self.K.shape) # 7819
            self.dx2 = np.random.rand(int(self.M - np.round(self.M /2)))*self.range +100

        # distances
        self.dx1 = np.random.rand(int(np.round(self.M /2)))*self.range -self.range  #[-range,0]
        self.dx = np.concatenate((self.dx1, self.dx2)) # horizontal size of a single RIS element
        self.dy = np.random.rand(self.M )*20-10 # vertical size of a single RIS element
        self.d_UR = ((self.dx - self.RIS[0])**2 + (self.dy - self.RIS[1])**2 + self.RIS[2]**2
                     )**0.5 # distance between centre and RIS
        self.d_RB = np.linalg.norm(self.BS - self.RIS) # distance between RIS and PS
        self.d_RIS = self.d_UR + self.d_RB # distance between the device and the RIS
        self.d_direct = ((self.dx - self.BS[0])**2 + (self.dy - self.BS[1])**2 + self.BS[2]**2
                         )**0.5  # distance between device and PS
        self.PL_direct = self.BS_Gain * self.User_Gain*(3*10**8/self.fc/4/np.pi/self.d_direct)**self.alpha_direct # path loss of direct path
        self.PL_RIS = self.BS_Gain*self.User_Gain*self.RIS_Gain*self.L**2*self.d_RIS**2/4/np.pi\
                    *(3*10**8/self.fc/4/np.pi/self.d_UR )**2*(3*10**8/self.fc/4/np.pi/self.d_RB)**2 # path loss of the cascaded devices-RIS-PS channel
        #channels
        h_d = (np.random.randn(self.N, self.M) + 1j*np.random.randn(self.N, self.M))/2**0.5
        h_d = h_d@np.diag(self.PL_direct**0.5)/self.ref
        H_RB = (np.random.randn(self.N, self.L) + 1j*np.random.randn(self.N, self.L))/2**0.5
        h_UR = (np.random.randn(self.L, self.M) + 1j*np.random.randn(self.L, self.M))/2**0.5
        h_UR = h_UR@np.diag(self.PL_RIS**0.5)/self.ref
        

        # collection j-th cascaded channel coefficient matrix for RIS
        G=np.zeros([self.N, self.L, self.M], dtype=complex)
        for j in range(self.M):
            G[:,:,j] =  H_RB @ np.diag(h_UR[:,j])

        # in this case, we consider h_direct only
        gamma = [15] # 15dB yields the best performance (RIS-FL)
        obj_new_NORIS, x_store, f_store = DC_NORIS(self.N, self.M, self.K, self.sigma, h_d, gamma, self.verbose)
        self.DC_NORIS_set[0] = obj_new_NORIS[0]

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))


            indices, selected_clients = self.select_clients(i, num_clients=self.M)  # uniform sampling
            #selected_clients = self.select_clients_convert(i, x_store.reshape(-1))
            np.random.seed(i) # random seed related to i
            # client selection - random
            # active_clients = np.random.choice(selected_clients, round(self.M * (1-self.drop_percent)), replace=False)
            # loss-ranked client selection
            # ids, gps, ns, tc, ls = self.train_error_and_loss() # no model exchange, GeometryMedian pending
            if i < 1:
                active_clients = np.random.choice(selected_clients, round(self.M * (1 - self.drop_percent)), replace=False).tolist() # first round - random selection
            else:
                drop_list = get_loss_rank[round(self.M * (1 - self.drop_percent))-1:]

                active_clients = list(filter(lambda x: x.id not in drop_list, selected_clients))
                #print("round:", i, active_clients, drop_list, len(active_clients)+len(drop_list))

            hf_store = {} # store channel matrix
            hf_store['x_stored'] = copy.deepcopy(x_store) # store device selection
            hf_store['f_stored'] = copy.deepcopy(f_store) # store normalized receiver beamforming vector with norm2(f)=1 # format: complex(a i + b j)
            hf_store['h_stored'] = copy.deepcopy(h_d)     # store channel selection

            csolns = []  # buffer for receiving client solutions
            closses = {} # buffer for receiving client losses
            for idx, (x_flag, c) in enumerate(zip(x_store, active_clients)):
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, shares, closs = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from active clients
                if x_flag:
                    csolns.append(soln) # (num_samples, soln)
                # gather local losses from client
                closses[c.id] = closs # closs collection TODO
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=shares, closs=closs) # (stats: bytes_w, comp, bytes_r)

            get_loss_rank = [*dict(sorted(closses.items(), key=lambda item: item[1]))] # unpacking with * eq to list(dict.keys())
            #print("get_loss_rank", get_loss_rank)
            # update models
            #self.latest_model = self.aggregate(csolns) # note: can introduce weighting factor for the "lazy" nodes
            self.latest_model = self.aggregate_aircomp(csolns, shares, hf_store) # parameters
            # aggregation error - 
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
