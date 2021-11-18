"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn

from spotlight.layers import ScaledEmbedding, ZeroEmbedding

import torch.nn.functional as F

from spotlight.torch_utils import gpu

import torch

import numpy as np


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.cf1 = nn.Linear(len(user_embedding_layer), 128)
            #why dont we put the star in front of the n_actions??
            self.cf2 = nn.Linear(128, self.embedding_dim)

            layer1 = F.relu(self.cf1(user_embedding_layer))

            self.user_embeddings = self.cf2(layer1)

        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias

    def get_state_embeddings(self, user_ids, num_items, item_ids=None, use_cuda = False):

        #self._check_input(user_ids, item_ids, allow_items_none=True)
        #self._net.train(False)


        if np.isscalar(user_ids):
            user_ids = np.array(user_ids, dtype=np.int64)


        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        user_var = gpu(user_ids, use_cuda)

        user_ids = user_var.squeeze()

        user_embedding = self.user_embeddings(user_ids)
        user_embedding = user_embedding.squeeze()
        return(user_embedding)

    def get_action_embeddings(self, item_ids, use_cuda = False):

        if np.isscalar(item_ids):
            item_ids = np.array(item_ids, dtype=np.int64)

        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        item_var = gpu(item_ids, use_cuda)

        item_ids = item_var.squeeze()

        item_embedding = self.item_embeddings(item_ids)
        item_embedding = item_embedding.squeeze()

        return(item_embedding)
