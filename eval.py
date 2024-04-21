import torch

def evaluate(generator, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_output_list = []
    eval_output_list = []
    original_output_list = []
    with torch.no_grad():
        for real_inputs, target_outputs in dataloader:
            real_inputs = real_inputs.to(device)
            target_outputs = target_outputs.to(device)
            real_inputs = real_inputs.to(device).float()
            target_outputs = target_outputs.to(device).float()

            real_inputs /= 127.5
            real_inputs -= 1
            eval_output = generator(real_inputs)
            num_sub_tensors = eval_output.size(0)
            for i in range(num_sub_tensors):
                target_output_list.append(target_outputs[i])
                eval_output_list.append(eval_output[i])
                original_output_list.append(real_inputs[i])
    return original_output_list,target_output_list,eval_output_list