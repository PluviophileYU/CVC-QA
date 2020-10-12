import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel
from model import BertForMultipleChoice_CVC, Jensen_Shannon_Div
from transformers import PreTrainedModel
WEIGHTS_NAME = "pytorch_model.bin"

class Post_MV(BertPreTrainedModel):
    def __init__(self, args, config):
        super(Post_MV, self).__init__(config)
        self.args = args
        self.config = config
        self.pre_model = BertForMultipleChoice_CVC.from_pretrained(args.checkpoint)
        self.config_class = self.pre_model.config_class

        for name, p in self.pre_model.named_parameters():
            p.requires_grad = False
        self.TIE = MV(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                input_ids_np=None, attention_mask_np=None, token_type_ids_np=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        outputs = self.pre_model.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        logits = self.pre_model.classifier(pooled_output).view(-1, num_choices)

        input_ids_np = input_ids_np.view(-1, input_ids_np.size(
            -1)) if input_ids_np is not None else None
        attention_mask_np = attention_mask_np.view(-1, attention_mask_np.size(
            -1)) if attention_mask_np is not None else None
        token_type_ids_np = token_type_ids_np.view(-1, token_type_ids_np.size(
            -1)) if token_type_ids_np is not None else None
        outputs_np = self.pre_model.bert(input_ids_np,
                               attention_mask=attention_mask_np,
                               token_type_ids=token_type_ids_np,
                               position_ids=position_ids,
                               head_mask=head_mask)
        outputs_np_ = outputs_np[2][self.pre_model.num_shared_layers - 1]
        logits_np = self.pre_model.MLP(outputs_np_, attention_mask_np)
        logits_np = self.pre_model.classifier_(logits_np).view(-1, num_choices)

        TIE_logits, item_1, item_2 = self.TIE(logits, logits_np)
        loss_fct = CrossEntropyLoss()
        TIE_loss = loss_fct(TIE_logits, labels)
        return TIE_logits, TIE_loss, item_1, item_2

class MV(nn.Module):
    def __init__(self, config):
        super(MV, self).__init__()
        self.config = config
        self.num_options = config.num_options
        self.compute_c_1 = nn.Linear(self.num_options*2+1, 100)
        self.compute_c_2 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()

    def compute_c(self, x):
        x = self.compute_c_1(x)
        x = self.tanh(x)
        x = self.compute_c_2(x)
        x = F.sigmoid(x)
        return x

    def forward(self, logit, logit_np):
        prob_zk = F.softmax(logit, dim=-1)
        prob_zb = F.softmax(logit_np, dim=-1)
        num_choice = prob_zk.size()[-1]
        js = Jensen_Shannon_Div(prob_zk, prob_zb).unsqueeze(-1)
        c = torch.cat([prob_zk, prob_zb, js], dim=-1)
        # c = torch.cat([logit, logit_np, logit-logit_np, js], dim=-1)
        c = self.compute_c(c)
        TIE_logits = prob_zk * prob_zb - c * prob_zb
        return TIE_logits, prob_zk * prob_zb, c * prob_zb