from transformers import LlamaForCausalLM as TLlamaForCausalLM
from transformers import LlamaPreTrainedModel as TLlamaPreTrainedModel
from attention_sinks.inject_mixin import InjectAttentionSinksMixin

class LlamaPreTrainedModel(InjectAttentionSinksMixin, TLlamaPreTrainedModel):
    pass

class LlamaForCausalLM(LlamaPreTrainedModel, TLlamaForCausalLM):
    pass