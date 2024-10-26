import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, *args, **kwargs):
        super(GMF, self).__init__(*args, **kwargs)

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weights_()

    def _init_weights_(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_user.weight, std=0.01)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        h1 = embed_user * embed_item
        prediction = self.predict_layer(h1)
        return prediction.view(-1)


class NeuMF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, mlp_layers, dropout, *args, **kwargs
    ):
        super(NeuMF, self).__init__(*args, **kwargs)

        self.dropout = dropout
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)

        assert (
            mlp_layers[-1] == factor_num
        ), "Final layer in MLP should have dimension = factor_num"
        assert (
            mlp_layers[0] % 2 == 0
        ), "MLP first layer should be even as the outputs of MLP Embeddings are concatenated"
        self.embed_user_MLP = nn.Embedding(user_num, mlp_layers[0] // 2)
        self.embed_item_MLP = nn.Embedding(item_num, mlp_layers[0] // 2)

        mlp_modules = []
        num_layers = len(mlp_layers)
        for i in range(num_layers - 1):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)

        predict_size = 2 * factor_num
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weights_()

    def _init_weights_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
