import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


class PretrainGeometricAttention(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim):
        super().__init__()

        self.node_dim = node_dim
        self.n_head = n_head
        self.head_dim = node_dim // n_head

        # * to alpha
        self.query = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.key = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.value = torch.nn.Linear(node_dim, self.head_dim * n_head, bias=False)
        self.pair2alpha = torch.nn.Linear(pair_dim, n_head, bias=False)
        self.conv2dalpha = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head * 2), torch.nn.Conv2d(n_head * 2, n_head, 3, 1, 1), torch.nn.LeakyReLU())

        # output
        self.out_transform = torch.nn.Sequential(torch.nn.LayerNorm(n_head * pair_dim + node_dim), torch.nn.Linear(n_head * pair_dim + node_dim, node_dim * 2), torch.nn.LayerNorm(node_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim * 2, node_dim))
        self.layer_norm = torch.nn.LayerNorm(node_dim)
        self.alpha2pair = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + pair_dim), torch.nn.Conv2d(n_head + pair_dim, pair_dim, 3, 1, 1), torch.nn.LeakyReLU())

    @staticmethod
    def _heads(x, n_head, n_ch):

        # x = [..., n_head * n_ch] -> [..., n_head, n_ch]
        s = list(x.shape)[:-1] + [n_head, n_ch]
        return x.view(*s)

    def _node2alpha(self, x):
        query_l = self._heads(self.query(x), self.n_head, self.head_dim)
        key_l = self._heads(self.key(x), self.n_head, self.head_dim)

        # query_l = [N, L, n_head, head_dim]
        # key_l = [N, L, n_head, head_dim]

        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)

        # query_l = [N, n_head, L, head_dim]
        # key_l = [N, n_head, head_dim, L]

        alpha = torch.matmul(query_l, key_l) / torch.sqrt(torch.FloatTensor([self.head_dim]).to(x.device))
        alpha = alpha.permute(0, 2, 3, 1)
        return alpha

    def _pair2alpha(self, z):
        alpha = self.pair2alpha(z)
        return alpha

    def _node_aggregation(self, alpha, x):
        N = x.shape[0]
        value_l = self._heads(self.value(x), self.n_head, self.head_dim)

        # value_l = [N, L, n_head, head_dim]

        value_l = value_l.permute(0, 2, 1, 3)

        # value_l = [N, n_head, L, head_dim]

        x = torch.matmul(alpha.permute(0, 3, 1, 2), value_l)

        # x = [N, n_head, L, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [N, L, n_head, head_dim]

        x = x.view(N, -1, self.node_dim)

        # x = [N, L, node_dim]

        return x

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        node_from_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)

        # node_from_pair = [N, L, L, n_head, pair_dim]

        node_from_pair = node_from_pair.sum(dim=2).reshape(N, L, -1)

        # node_from_pair = [N, L, n_head * pair_dim]

        return node_from_pair

    def forward(self, x, z, mask):

        # x = [N, L, node_dim]
        # z = [N, L, L, pair_dim]
        # mask = [N, L]

        alpha_from_node = self._node2alpha(x)
        alpha_from_pair = self._pair2alpha(z)

        # alpha_from_node = [N, L, L, n_head]
        # alpha_from_pair = [N, L, L, n_head]

        alpha_sum = self.conv2dalpha(torch.cat((alpha_from_pair, alpha_from_node), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        N, L = alpha_sum.shape[:2]
        mask_row = mask.view(N, L, 1, 1).expand_as(alpha_sum)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
        alpha_sum = torch.where(mask_pair, alpha_sum, alpha_sum - 1e6)
        alpha = torch.softmax(alpha_sum, dim=2)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))

        # alpha = [N, L, L, n_head]

        node_from_node = self._node_aggregation(alpha, x)
        node_from_pair = self._pair_aggregation(alpha, z)

        # node_from_node = [N, L, node_dim]
        # node_from_pair = [N, L, n_head * pair_dim]

        x_out = self.out_transform(torch.cat([node_from_pair, node_from_node], dim=-1))
        x = self.layer_norm(x + x_out)
        return x, self.alpha2pair(torch.cat((z, alpha), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class PretrainEncoder(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.pair_encoder = torch.nn.Linear(7, pair_dim)
        self.blocks = torch.nn.ModuleList([PretrainGeometricAttention(node_dim, n_head, pair_dim) for _ in range(num_layer)])

    def forward(self, embedding, pair, atom_mask):

        # embedding = [N, L, 1280]
        # pair = [N, L, L, 7]
        # atom_mask = [N, L, 14]

        embedding = self.esm2_transform(embedding)

        # embedding = [N, L, node_dim]

        pair = self.pair_encoder(pair)

        # pair = [N, L, L, pair_dim]

        for block in self.blocks:
            embedding, pair = block(embedding, pair, atom_mask[:, :, ATOM_CA])

        return embedding


class PretrainModel(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.pretrain_encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.pretrain_mlp = torch.nn.Linear(node_dim, 20)
        self.logits_coef = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))

    def forward(self, data):

        # add plddt gate
        plddt = torch.sign(torch.relu(data["fixed_embedding"][:, :, -1] - 0.7)).bool()
        data["atom_mask"] = torch.stack((data["atom_mask"], plddt.unsqueeze(-1).repeat(1, 1, 14)), dim=0).all(dim=0)

        node_feat = self.pretrain_encoder(data["dynamic_embedding"], data["pair"], data["atom_mask"])

        logits_pred = self.pretrain_mlp(node_feat)
        logits_pred = self.logits_coef[None, :, None, None] * torch.cat((logits_pred.unsqueeze(1), data["logits"]), dim=1)
        logits_pred = logits_pred.mean(1)
        return logits_pred
