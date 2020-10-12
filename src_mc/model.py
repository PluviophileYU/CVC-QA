from transformers.modeling_bert import BertPreTrainedModel, BertForMultipleChoice, BertModel, BertLayer, BertConfig, gelu, BertPooler
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import copy

class Fusion(nn.Module):
    def __init__(self,
                 fusion_name, config):
        super(Fusion, self).__init__()
        FUSION_METHOD = {
            'prob_rubi': self.prob_rubi_fusion
        }
        self.fusion_method = FUSION_METHOD[fusion_name]

    def prob_rubi_fusion(self, zk, zb, hidden=None, cf=False):
        if not cf:
            prob_zk = F.softmax(zk, dim=-1)
            prob_zb = F.softmax(zb, dim=-1)
            fusion_prob = prob_zk * prob_zb
            log_fusion_prob = torch.log(fusion_prob)
            return log_fusion_prob
        else:
            prob_zk = F.softmax(zk, dim=-1)
            prob_zb = F.softmax(zb, dim=-1)
            num_choice = prob_zk.size()[-1]
            similarity = Jensen_Shannon_Div(prob_zk, prob_zb).unsqueeze(-1)
            # c = torch.mean(prob_zk, dim=-1).unsqueeze(-1).repeat(1, num_choice)
            c = similarity
            log_fusion_prob = prob_zk * prob_zb - c * prob_zb
            return log_fusion_prob, c

    def forward(self, zk, zb, hidden=None, cf=False):
        return self.fusion_method(zk, zb, hidden, cf)

class BertForMultipleChoice_CVC(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice_CVC, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_ = nn.Linear(config.hidden_size, 1)
        if config.num_hidden_layers == 12:
            self.num_shared_layers = 10
        elif config.num_hidden_layers == 24:
            self.num_shared_layers = 20
        self.fusion_name = 'prob_rubi'
        self.Fusion = Fusion(self.fusion_name, config)
        self.MLP = transformer_block(config, num_shared_layers=self.num_shared_layers, num_layers=config.num_hidden_layers-self.num_shared_layers)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                input_ids_np=None, attention_mask_np=None, token_type_ids_np=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).view(-1, num_choices)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            input_ids_np = input_ids_np.view(-1, input_ids_np.size(
                -1)) if input_ids_np is not None else None
            attention_mask_np = attention_mask_np.view(-1, attention_mask_np.size(
                -1)) if attention_mask_np is not None else None
            token_type_ids_np = token_type_ids_np.view(-1, token_type_ids_np.size(
                -1)) if token_type_ids_np is not None else None

            outputs_np = self.bert(input_ids_np,
                                attention_mask=attention_mask_np,
                                token_type_ids=token_type_ids_np,
                                position_ids=position_ids,
                                head_mask=head_mask)

            # Use Transformer block on top of bias branch
            outputs_np = outputs_np[2][self.num_shared_layers - 1]
            outputs_np = grad_mul_const(outputs_np, 0.0)
            logits_np = self.MLP(outputs_np, attention_mask_np)
            logits_np = self.classifier_(logits_np).view(-1, num_choices)

            logits_np_ = logits_np*1
            fusion_logits = self.Fusion(logits, logits_np_.detach())

            loss_fct = CrossEntropyLoss()
            fusion_loss = loss_fct(fusion_logits, labels)
            np_loss = loss_fct(logits_np, labels)


            outputs = (fusion_loss, np_loss) + outputs
        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def inference_IV(self, input_ids, attention_mask=None, token_type_ids=None,
                input_ids_np=None, attention_mask_np=None, token_type_ids_np=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output).view(-1, num_choices)
        return logits

class transformer_block(nn.Module):
    def __init__(self, config, num_shared_layers, num_layers):
        super(transformer_block, self).__init__()
        self.num_layers = num_layers
        self.num_shared_layers = num_shared_layers
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(num_layers)])
        self.pooler = BertPooler(config)

    def copy_from_bert(self, bert):
        for i, layer in enumerate(self.bert_layers):
            self.bert_layers[i] = copy.deepcopy(bert.encoder.layer[i+self.num_shared_layers])
        self.pooler = copy.deepcopy(bert.pooler)

    def forward(self, x, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for bert_layer in self.bert_layers:
            x = bert_layer(x, extended_attention_mask)
            x = x[0] # only the hidden part, bert layer will also output attention
        x = self.pooler(x)
        return x

class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)

def logits_norm(x):
    return F.softmax(x, dim=-1)

def Jensen_Shannon_Div(p1, p2):
    batch_size = p1.size()[0]
    result = []
    for i in range(batch_size):
        p1_, p2_ = p1[i], p2[i]
        JS_div = 0.5*KL_Div(p1_, (p1_+p2_)/2) + 0.5*KL_Div(p2_, (p1_+p2_)/2)
        result.append(JS_div.unsqueeze(0))
    output = torch.cat(result, dim=0)
    return output


def KL_Div(P, Q):
    output = (P * (P / Q).log()).sum()
    return output