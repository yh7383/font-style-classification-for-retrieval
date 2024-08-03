import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss



class NpairsLossTest():
    def testNpairs(self):
        num_data = 16
        feat_dim = 5
        num_classes = 3
        reg_lambda = 0.02

        embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)
        embeddings_positive = np.random.rand(num_data, feat_dim).astype(np.float32)

        labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

        # Reshape labels to compute adjacency matrix.
        labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

        # Compute the loss in pytorch
        npairloss = NpairLoss()
        loss_tc = npairloss(
            anchor=torch.tensor(embeddings_anchor),
            positive=torch.tensor(embeddings_positive),
            target=torch.from_numpy(labels)
        )

        print('pytorch version: ', loss_tc.numpy())
        # self.assertAllClose(loss_np, loss_tf)


if __name__ == '__main__':
    NpairsLossTest().testNpairs()