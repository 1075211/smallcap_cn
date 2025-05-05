from transformers import PreTrainedTokenizer
from sentencepiece import SentencePieceProcessor

class MengziTokenizer(PreTrainedTokenizer):
    def __init__(self, model_file, **kwargs):
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_file)
        super().__init__(**kwargs)
    
    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)
    
    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)
    
    # 实现其他必要方法...
