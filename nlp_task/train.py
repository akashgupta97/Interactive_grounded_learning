import nlp_data_preproc
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import wandb
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


class CommandsDataset(Dataset):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def __len__(self):
        return len(self.inputs)


def train(model, train_loader, tokenizer, history_len):

    val_input, val_output = nlp_data.prepare_data('val_data_part1.json', history_len=None)
    val_dataset = CommandsDataset(val_input, val_output)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, drop_last=True, 
                              collate_fn=lambda x : nlp_data_preproc.prepare_inputs(x, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    n_epochs = 1
    min_avg_loss = 1e9
    task_prefix = "implement given instructions: "


    for epoch in range(n_epochs):
        for i, (ids, mask, labels) in enumerate(train_loader):
            # forward pass
            loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss
            #losses.append(loss.item())
            wandb.log({"train_loss" : loss.item()}, step=i)
            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 200 == 199:
                model.eval()
                val_losses = []
                for ids, mask, labels in val_loader:
                    with torch.no_grad():
                        loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss
                        wandb.log({"val_loss" : loss.item()})
                        val_losses.append(loss.item())

                if sum(val_losses)/len(val_losses) < min_avg_loss:
                    min_avg_loss = sum(val_losses)/len(val_losses)
                    torch.save(model.state_dict(), f't5-autoregressive-history-{history_len}-best.pt')

                test_examples = np.random.choice(len(val_dataset), 10)

                text_table = wandb.Table(columns=["inputs", "predictions", "ground_truth", "BLEU"])
                for idx in test_examples:
                    with torch.no_grad():
                        inputs = f"{task_prefix}{val_input[idx]}"
                        input_ids = tokenizer(f"{task_prefix}{val_input[idx]}", return_tensors="pt").input_ids

                        outputs = model.generate(input_ids.cuda(), min_length=2, max_length=128)
                        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)

                        # BLEU is a metric that measures the similarity between the candidate sentence produced by 
                        # a machine translation system and a reference sentence, by computing the overlap between n-grams
                        bleu_score = sentence_bleu([tokenizer.tokenize(val_output[idx])], tokenizer.tokenize(outputs))
                        text_table.add_data(inputs, outputs, val_output[idx], bleu_score)

                wandb.log({"validation_samples" : text_table})
                model.train()

    torch.save(model.state_dict(), f't5-autoregressive-history-{history_len}-last.pt')

def main():
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    max_source_length = 512
    max_target_length = 128
    special_tokens_dict = {'additional_special_tokens': ['<Architect>', '<Builder>', '<sep1>']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    history_len = None

    wandb.init(project='text2action', name=f'autoregressive_history_{history_len}')
    
    train_inputs, train_outputs = nlp_data_preproc.prepare_data('train_data_augmented_part1.json', history_len=history_len)
    train_dataset = CommandsDataset(train_inputs, train_outputs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, 
                                collate_fn=lambda x : nlp_data_preproc.prepare_inputs(x, tokenizer, max_source_length, max_target_length))
    
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()

    train(model, train_loader, tokenizer, history_len)


if __name__ == '__main__':
    sys.exit(main())
