from os import stat
from tkinter import W
import copy
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        # create worker nodes
        tf.compat.v1.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        #super(BaseFedarated, self).__init__(params) # 06/Sep/2022 added

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset # tbd: remove empty files
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() # tol, loss, sample_size
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def select_clients_convert(self, round, x_clients):
        # x_client: consider channel capacity at this stage.
        num_clients = min(np.sum(x_clients).astype(int), len(self.clients))
        np.random.seed(round)
        # map to the indices
        #indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        
        return np.asarray(self.clients)[x_clients]

    def aggregate(self, wsolns):
        total_weight = 0.0
        
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def aggregate_aircomp(self, wsolns, stats, dict_store):
        """ aggregation transmission
        wsolns: [(weight, solution) * num_clients] NOTE mclr with ONLY one layer w(60,10) b(10,)
        stats: shared states including 
        dict_store: dict obj where dict_store.keys()=['x_stored', 'h_stored', 'f_stored']

        Returns:
        averaged_soln: 
        """

        ds = int(stats[0] / wsolns[0][1][1].dtype.itemsize) # size of graph in bytes / byte_size
        
        index = (dict_store['x_stored']==1).squeeze() # if the communication was successful  (used in transmission)
        
        N = self.N
        K = self.K[index] # num of trainng samples
        K2 = K**2
        h = dict_store['h_stored'] # h_direct 
        f = dict_store['f_stored'].squeeze() # normalized receiver beamforming vector with norm2(f)=1 NOTE scipy ver being too high
        
        inner = f.conj()@h[:,index] # matmul 
        inner2 = np.abs(inner)**2

        total_weight = 0.0
        # len(wsolns[0]), len(wsolns[0][1]), wsolns[0][1][0].shape, wsolns[0][1][1].shape # 2 2 (60,10) (10,)
        base = [0] * len(wsolns[0][1]) # len of soln
        ws = []
        grad = []
        for (w, soln) in wsolns: # weighted solutions
            total_weight += w
            g = copy.deepcopy(soln) # NOTE: grad=w.reshape([1,-1])
            grad.append(np.vstack([g[0], g[1].reshape(1,-1)]).reshape(-1))
            ws.append(w)
        grad_mean = np.mean(grad, axis=1)
        #print("grad max min mean", np.max(grad, axis=1), np.min(grad, axis=1), grad_mean)
        grad_var = np.clip(np.var(grad, axis=1), -10, 10)
        var_sqrt = grad_var**0.5
        
        grad_bar = np.asarray(ws)@grad_mean # takes (n,k) @ (k,m) -> (n,m)

        eta = np.min(self.transmitPower * inner2 / grad_var) # increases signal transPower
        eta_sqrt = eta**0.5
            
        b = np.asarray(ws) * eta_sqrt * var_sqrt * inner.conj() / inner2

        noise_power = self.sigma * self.transmitPower 
        # random noise on antennas
        rand_n = (np.random.randn(N, ds) + 1j*np.random.randn(N, ds)) / (2)**0.5*noise_power**0.5
        # (610, indices.sum()) 
            
        x_sig = np.tile(b/var_sqrt, (ds, 1)).T * (np.asarray(grad) - np.tile(grad_mean, (ds, 1)).T)
        #x_sig = np.asarray(ws).reshape(-1, 1) * x_sig
        y = h[:, index]@x_sig + rand_n

        y_received = np.real((f.conj()@y/eta_sqrt+grad_bar))/ total_weight # np.asarray(ws)
        
        # reshape and return
        #for i, v in enumerate(wsoln):       base[i] += w * v.astype(np.float64) 
        base[0] += y_received[:-10].reshape(wsolns[0][1][0].shape)
        base[1] += y_received[-10:].reshape(wsolns[0][1][1].shape)

        averaged_soln = [v / 1.0 for v in base] # total_weight

        return averaged_soln
