import torch

def get_ordereddict_size(ordered_dict) -> int:
    total_bytes = 0
    first_weight_processed=False
    for key, values in ordered_dict.items():
        if "weight" in key and not first_weight_processed:
            for i in range(len(values)):
                tensor = values[i]
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
                else:
                    for j in range(len(tensor)):
                        total_bytes += tensor[j].numel() * tensor[j].element_size()

            first_weight_processed = True
        # if isinstance(values, torch.Tensor):
        #     total_bytes += values.numel() * values.element_size()
        # else:
            # for i in range(len(values)):
            #     tensor = values[i]
            #     # tensor = values[0]
            #     if isinstance(tensor, torch.Tensor):
            #         total_bytes += tensor.numel() * tensor.element_size()
            #     else:
            #         for j in range(len(tensor)):
            #             total_bytes += tensor[j].numel() * tensor[j].element_size()

    return total_bytes

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"