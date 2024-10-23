import torch
import numpy as np
from datasets import load_dataset

class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split='train', max_seq_len=512):
        assert split in ['train', 'test']
        gsm8k = load_dataset(path='gsm8k', name='main')
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.data = gsm8k[split].filter(
            lambda row: self._filter_on_max_seq_len(row, tokenizer, max_seq_len)
        )

    @staticmethod
    def _filter_on_max_seq_len(row, tokenizer, max_seq_len):
        # GSM8K doc_to_text: "Question: {{question}}\nAnswer:"
        prompt_tokens = tokenizer.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=True)
        response_tokens = tokenizer.encode(row['answer'], add_special_tokens=False)
        response_tokens.append(tokenizer.eos_token_id)
        return len(prompt_tokens) + len(response_tokens) <= max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns necessary inputs for transformers.LlamaForCausalLM forward
        - input_ids: indices of input sequence tokens in the vocabulary
        - attention_mask: mask to avoid performing attention on padding token indices (0 for tokens that are masked)
        - labels: of the same shape as input_ids, with -100 in place of masked tokens and question tokens
        """
        row = self.data[idx]
        input_ids = np.zeros(self.max_seq_len, dtype=int)
        attention_mask = np.zeros(self.max_seq_len, dtype=int)
        labels = np.full(self.max_seq_len, fill_value=-100, dtype=int)

        # GSM8K doc_to_text: "Question: {{question}}\nAnswer:"
        prompt_tokens = self.tokenizer.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=True)
        response_tokens = self.tokenizer.encode(row['answer'], add_special_tokens=False)
        response_tokens.append(self.tokenizer.eos_token_id)

        prompt_len = len(prompt_tokens)
        response_len = len(response_tokens)
        assert prompt_len + response_len <= self.max_seq_len, f"Exceeded max sequence length ({self.max_seq_len}) with: {prompt_len + response_len}"

        input_ids[:(prompt_len + response_len)] = prompt_tokens + response_tokens
        # attention_mask: attend to all input tokens
        attention_mask[:(prompt_len + response_len)] = 1
        # labels: -100 for question tokens, 0 for answer tokens
        labels[prompt_len:prompt_len + response_len] = response_tokens

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
        
def get_dataloader(tokenizer, batch_size, split, shuffle, max_seq_len=1024, num_workers=4):
    dataset = GSM8KDataset(tokenizer, split=split, max_seq_len=max_seq_len)
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       pin_memory=True, 
                                       drop_last=True)