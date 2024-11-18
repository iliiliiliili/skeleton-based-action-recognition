import os
import numpy as np
import matplotlib.pyplot as plt


def draw_uncertain_attentions(attentions: dict, path: str, aggregation=None, cmap="Greens", names=["Identity", "Inward", "Outward"]):
    
    figure, axarr = plt.subplots(
        len(attentions.items()),
        2 * 3,
        figsize=(15 * 2 * 3, 15 * len(attentions.items())),
    )

    def aggregate(input):
        if aggregation == "mean":
            return input.mean(axis=0)
        if aggregation == "first":
            return input[0, :, :]
        if aggregation is None:
            return input
        
        raise ValueError()


    for i, (key, atts) in enumerate(attentions.items()):
        
        att, att_var = atts
        att_var = att_var / (att.abs() + 1e-7)
        att = att.detach().cpu().numpy().squeeze()
        att_var = att_var.detach().cpu().numpy().squeeze()

        for q in range(0, 3):
        
            ax = axarr[i, 0 + q * 2]
            data = aggregate(att[q])
            cax = ax.imshow(data, cmap=cmap)
            ax.set_title(f"Attention[{names[q]}] for layer {key}")
            cbar = figure.colorbar(cax, orientation='horizontal')

            ax = axarr[i, 1 + q * 2]
            data = aggregate(att_var[q])
            cax = ax.imshow(data, cmap=cmap)
            ax.set_title(f"Attention[{names[q]}] uncertainty for layer {key}")
            cbar = figure.colorbar(cax, orientation='horizontal')

    figure.savefig(path)

    print()
