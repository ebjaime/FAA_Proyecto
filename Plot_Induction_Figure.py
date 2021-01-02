#####
#
#
#
#
# In the paper, we have used inductions of July 23rd. Looking at the
# metadata, the id of the presentations are 17 (banana) and 19 (wine).
#
#####

## Importing libraries
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.gridspec as gridspec

id = 38

def graficaInduccion(id=38, metadata=None, dataset=None):
    if metadata is None:
        columns_metadata = ["id", "date", "class", "t0", "dt"]
        ## Importing metadata and induction information
        metadata = np.loadtxt('data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
    metadata_aux = np.array( metadata[:,[0,3,4]], dtype=float )
    if dataset is None:
        columns_dataset = ["id","time", "R1","R2","R3","R4","R5","R6","R7","R8", "Temp.", "Humidity"]
        dataset = np.loadtxt('data/HT_Sensor_dataset.dat', skiprows=1)

    # df_metadata = pd.DataFrame(data=metadata, columns=columns_metadata)

    info = metadata_aux[id]
    t0 = info[1]
    tf = t0 + info[2]


    ## Loading the dataset

    # df_dataset = pd.DataFrame(data=dataset, columns=columns_dataset)

    Data = dataset[dataset[:,0] == id,1:]
    Data[:,0] += t0




    ## Starting the plot

    pl.figure(figsize = (9,7))
    gs1 = gridspec.GridSpec(6, 1  )
    gs1.update(wspace=0.4, hspace=0.0)

    ax = {}
    for j in range(6):
        ax[j,0] = pl.subplot(gs1[j])
        # ax[j,1] = pl.subplot(gs1[j*2+1])



    ## Plotting all data
    ax[0,0].plot( Data[:,0], Data[:,10], color=(0.1,0.8,0.1), lw=1.5 )
    ax[0,0].set_ylim(min(Data[:,10])-5, max(Data[:, 10])+5)

    ax[1,0].plot( Data[:,0], Data[:, 9], color=(1.0,0.1,0.0), lw=1.5 )
    ax[1,0].set_ylim(min(Data[:,9])-5, max(Data[:, 9])+5)

    idxs = [[1,4], [2,3], [5,6], [7,8]]

    for i, r in enumerate(idxs):
        ax[2+i,0].plot( Data[:,0], Data[:, r[0]], color=(0.3,0.3,0.3), lw=1.5, label="R"+str(r[0]))
        ax[2+i,0].plot( Data[:,0], Data[:, r[1]], '-', color=(1.0,0.5,0.1), lw=1.5, label="R"+str(r[1]) )
        ax[2+i,0].set_ylim(min(np.minimum(Data[:,r[0]], Data[:,r[1]]))-0.5 , max(np.maximum(Data[:,r[0]], Data[:,r[1]]))+0.5 )
        # ax[2+i,0].set_yticks( [] )
        ax[2+i,0].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

    ## Setting up limits and ticks

    for j in range(6):
        ax[j,0].plot( [t0,t0,tf,tf], [-100,100,100,-100], '-',
                      color=(0.1,0.1,1.0), lw=2.0, alpha = 0.5 )

        # ax[j,0].set_xticks([])
        ax[j,0].set_xlim(min(Data[:,0]), max(Data[:,0]))
        # ax[j,0].set_xticklabels([])




    ## Writing labels

    ax[0,0].set_title(metadata[id][2])

    ax[0,0].set_ylabel(r"$H$ (%)")
    ax[1,0].set_ylabel(r"$T_E$ (C)")
    ax[2,0].set_ylabel(r"$R_{1,4}$ (k$\Omega$)")
    ax[3,0].set_ylabel(r"$R_{2,3}$ (k$\Omega$)")
    ax[4,0].set_ylabel(r"$R_{5,6}$ (k$\Omega$)")
    ax[5,0].set_ylabel(r"$R_{7,8}$ (k$\Omega$)")


    # ax[5,0].set_xticklabels([])
    ax[5,0].set_xlabel("Time (h)")

if __name__ == "__main__":
    plotFigure()
    ## Saving the plot as a png file
    pl.savefig("Huerta_etal_2016_Figure7.png", dpi=300)
