# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from renom_tda import Lens

try:
    from renom.optimizer import Sgd
except:
    raise Exception("This lens require renom modules.")


class AutoEncoder(Lens):
    """Class of Auto Encoder dimention reduction lens.

    Params:
        epoch: training epoch.

        batch_size: batch size.

        opt: training optimizer.

        network: auto encoder network class.

        verbose: print message or not.
    """

    def __init__(self, epoch, batch_size, network, opt=Sgd(), verbose=0):
        self.epoch = epoch
        self.batch_size = batch_size
        self.network = network
        self.opt = opt
        self.verbose = verbose

    def fit_transform(self, data):
        """dimention reduction function.

        Params:
            data: raw data or distance matrix.
        """
        n = data.shape[0]

        train_data, test_data = train_test_split(data, test_size=0.1, random_state=10)

        for i in range(self.epoch):
            total_loss = 0

            for ii in range(int(n / self.batch_size)):
                batch = data[ii * self.batch_size: (ii + 1) * self.batch_size]

                with self.network.train():
                    loss = self.network(batch)
                    loss.grad().update(self.opt)

                total_loss += loss
            if self.verbose == 1:
                print("epoch:{} loss:{}".format(i, total_loss / (n / self.batch_size)))

        projected_data = self.network.encode(data)
        return projected_data
