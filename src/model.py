from torch import nn 
from transformers import AutoModelForPreTraining, AutoModel
import sys
import warnings
warnings.filterwarnings("ignore")

class LOTClassModel(nn.Module): 
    def __init__(self, hidden_size:int =768, num_labels:int=5, hidden_dropout_prob:float=0.25): 
        super(LOTClassModel, self).__init__()

        self.num_labels = num_labels 
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.cls = AutoModelForPreTraining.from_pretrained('bert-base-uncased').cls 
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, num_labels)
        #self.init_weights()
        for param in self.cls.parameters():
            param.requires_grad = False
        
        for param in self.bert.parameters():
            param.requires_grad = False

    
    def forward(self, input_ids, pred_mode,attention_mask=None, token_type_ids=None,
                      position_ids=None, head_mask=None, inputs_embeds=None): 
        bert_outputs = self.bert(input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        if pred_mode.lower() ==  "classification": 
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode.lower() == "mlm": 
            logits = self.cls(last_hidden_states, bert_outputs[1])
        else: 
            sys.exit("Wrong pred_mode!")
        
        return logits
        
        