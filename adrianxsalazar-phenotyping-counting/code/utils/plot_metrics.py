import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_metrics():
    met_df: pd.DataFrame = pd.read_pickle("metrics.pkl");

    met_df = met_df.assign(epoch=met_df.index+1);

    # Fix incorrect metric reporting in from old code
    print(met_df.columns.to_list())
    if ("val_mse" in met_df.columns.to_list()): met_df = met_df.rename( columns= { "val_mse": "val_mae" } );


    # met_df["epoch"] += 99;
    print(met_df);

    met_mdf = met_df.melt("epoch", var_name="metric", value_name="value")

    sns.set_theme();
    plt.figure(figsize=(16,9), dpi=72);
    ax = plt.subplot(2,2,1);
    ax.set_title("Validation trend after 9th epoch")
    sns.regplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="val_mae", marker="x", ax=ax);

    ax = plt.subplot(2,2,2);
    ax.set_title("Loss function trend after 9th epoch")
    sns.regplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="train_loss", marker="x", ax=ax);

    ax = plt.subplot(2,2,3);
    ax.set_title("Performance after 9th epoch")
    sns.lineplot(data=met_mdf.where(met_mdf.epoch >= 10), x="epoch", y="value", hue="metric", palette="flare", ax=ax);

    ax = plt.subplot(2,2,4);
    ax.set_title("Performance before 10th epoch")
    sns.lineplot(data=met_mdf.where(met_mdf.epoch < 10), x="epoch", y="value", hue="metric", palette="flare", ax=ax);

    plt.tight_layout();
    plt.savefig("metrics.png", dpi=300);
    plt.show();


plot_metrics();