import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_metrics():
    met_df: pd.DataFrame = pd.read_pickle("metrics.pkl");

    met_df = met_df.assign(epoch=met_df.index+1);

    # Fix incorrect metric reporting in from old code
    print(met_df.columns.to_list())
    # if ("val_mse" in met_df.columns.to_list()): met_df = met_df.rename( columns= { "val_mse": "val_mae" } );


    # met_df["epoch"] += 99;
    print(met_df);

    met_mdf = met_df.melt("epoch", var_name="metric", value_name="value")

    sns.set_theme();
    plt.figure(figsize=(16,9), dpi=72);
    # ax = plt.subplot(2,2,1);
    # ax.set_title("Validation trend after 9th epoch")
    # sns.regplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="val_mae", marker="x", ax=ax);

    # ax = plt.subplot(2,2,2);
    # ax.set_title("Loss function trend after 9th epoch")
    # sns.regplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="train_loss", marker="x", ax=ax);

    ax = plt.subplot(1,1,1);
    ax.set_title("Performance from 10th epoch")
    line_val  = ax.plot("epoch", "val_mse", label="val_mse", data=met_df.where(met_df.epoch >= 10), color="darkolivegreen");
    ax.set_ylabel("val_mae");
    # sns.lineplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="train_loss", ax=ax.twinx());
    axt = ax.twinx();
    line_loss = axt.plot("epoch", "train_loss", label="train_mse", data=met_df.where(met_df.epoch >= 10), color="salmon");
    axt.set_ylabel("loss_mae");
    lns  = line_val + line_loss;
    labs = [l.get_label() for l in lns];
    ax.legend(lns, labs);

    # ax = plt.subplot(1,2,2);
    # ax.set_title("Performance after 9th epoch")
    # line_val  = ax.plot("epoch", "val_mse", label="val_mse", data=met_df.where(met_df.epoch >= 10), color="darkolivegreen");
    # ax.set_ylabel("val_mse");
    # # sns.lineplot(data=met_df.where(met_df.epoch >= 10), x="epoch", y="train_loss", ax=ax.twinx());
    # axt = ax.twinx();
    # line_loss = axt.plot("epoch", "val_mae", label="val_mae", data=met_df.where(met_df.epoch >= 10), color="salmon");
    # axt.set_ylabel("val_mae");
    # lns  = line_val + line_loss;
    # labs = [l.get_label() for l in lns];
    # ax.legend(lns, labs);
    # axt.legend();

    # ax = plt.subplot(1,2,2);
    # ax.set_title("Performance before 10th epoch")
    # sns.lineplot(data=met_mdf.where(met_mdf.epoch < 10), x="epoch", y="value", palette="flare", ax=ax);

    plt.tight_layout();
    plt.savefig("metrics.png", dpi=300);
    plt.show();


plot_metrics();