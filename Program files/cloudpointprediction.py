import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from math import isnan
import datetime

import torch
from torch.nn import MSELoss
from torch.utils.data import random_split, Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import dataloader as dl
from parameters import RunParameters as Pm

dl.default_export_path()

sns.set_theme(style="white", palette=None)
print("LR: %s; BS: %s" % (Pm.LEARNING_RATE, Pm.BATCH_SIZE))


class TranTempPred(pl.LightningModule):

    def __init__(self):
        super().__init__()
        layer = []
        mom_layer_size = Pm.Layer_size
        while mom_layer_size >= 3:
            next_layer_size = int(mom_layer_size / 3)
            layer.append(torch.nn.Linear(mom_layer_size, next_layer_size))
            layer.append(torch.nn.ELU())
            mom_layer_size = next_layer_size

        if mom_layer_size != 1:
            layer.append(torch.nn.Linear(mom_layer_size, 1))

        print("%s layers are in use" % len(layer))
        self.nnl = torch.nn.Sequential(*layer)
        self.loss = MSELoss()

    def forward(self, x):
        # x = self.layer_1(x)
        # x = self.layer_2(x)
        # x = self.layer_3(x)
        x = self.nnl(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), Pm.LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)

        Pm.current_epoch_nr = self.current_epoch
        # # animation for display of learning curve
        # epoch = self.current_epoch
        # plt.plot([y.max(), y.min()], [y.max(), y.min()])
        # plt.plot(y_hat.detach().numpy, y, "o")
        # plt.ylabel(r"$Measured \ T_cp$")
        # plt.xlabel(r"$Predicted \ T_cp$")
        # plt.title(label="Training Epoch {}".format(epoch))
        # learning_animation_folder = os.path.join(project_folder,
        #                                          "LR{:.0E} Bs{} mE{} FpS{} Ld{}".format(LEARNING_RATE, BATCH_SIZE,
        #                                                                                 MAX_EPOCHS, FP_SIZE,
        #                                                                                 LAYER_DEPTHS))
        # os.makedirs(learning_animation_folder, exist_ok=True)
        # plt.savefig(fname=os.path.join(project_folder, "Epoch_{}".format(epoch)))
        # plt.close()
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss


class AddDataSet(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(np.array(df_edit["togeth"].to_list())).float()
        self.y_data = torch.tensor(df_edit["cloud_point"].values).float().unsqueeze_(-1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y


class DataManager(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = AddDataSet()
        split = (np.array([0.8, 0.1, 0.1]) * len(dataset)).astype(int).tolist()
        split[0] += len(dataset) - sum(split)
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, split, generator=torch.Generator().manual_seed(7))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


def print_plot(model, loss_function, save=False):
    # give out the whole dataset as a table with predicted over actual solubility, to make the error more vivid.
    plot_dataset = AddDataSet()

    y_hat = model(plot_dataset.x_data)
    plot_x = y_hat.detach().numpy()  # detach() to avoid changing the tensor in its original save
    plot_y = plot_dataset.y_data.numpy()
    loss = loss_function(y_hat, plot_dataset.y_data)

    Pm.last_loss = loss

    print("------- overall Loss {} -------".format(loss))
    # adding in an orientation line
    o_line = [[plot_y.max(), plot_y.min()], [plot_y.max(), plot_y.min()]]
    # the star solves iterable objects like lists, tuples, arrays, tensors after their first dimension
    #       eg. o_line = [[x,y], [x2,y2]] -> o_line_a, o_line_b = [x,y], [x2,y2]

    plt.plot(*o_line)
    plt.plot(plot_x, plot_y, "o")
    # early stopping with a patience of 50 means the final training epoch is
    # somewhat between the current and current - patience

    plt.title(label="Training with {}/{} Epochs, {}lr & {}Bs; loss is {:.2f}".format(
        Pm.current_epoch_nr, Pm.MAX_EPOCHS, Pm.LR, Pm.BATCH_SIZE, loss))

    plt.ylabel(r"Measured $T_{cp}$ [°C]")
    plt.xlabel(r"Predicted $T_{cp}$ [°C]")
    if save:
        plt.savefig(fname=os.path.join(Pm.export_path,
                                       "{} Epochs, {}FPs, {}Lr & {}Bs; {:.2f}loss.png".format(
                                           Pm.current_epoch_nr, Pm.FP_SIZE, Pm.LR,
                                           Pm.BATCH_SIZE, loss)))

    plt.show()
    plt.close()


def print_plot_2(loaded_model, loss_function, cur_epoch_nr, save=False, ):
    plot_dataset = AddDataSet()

    y_hat = loaded_model(plot_dataset.x_data)
    x = y_hat.detach().numpy()
    y = plot_dataset.y_data.numpy()
    loss = loss_function(y_hat, plot_dataset.y_data)
    o_line = [[y.max(), y.min()], [y.max(), y.min()]]
    plt.hexbin(x, y, gridsize=30, cmap="ocean_r", linewidths=0.01)
    plt.plot(*o_line, color="orange")
    # plt.title(label="Training with {}/{} Epochs; loss is {:.2f}".format(Pm.current_epoch_nr,
    # MAX_EPOCHS, loss))
    plt.colorbar()
    plt.ylabel(r"Measured $T_{cp}$ [°C]")
    plt.xlabel(r"Predicted $T_{cp}$ [°C]")
    if save:
        plt.savefig(
            fname=os.path.join(Pm.export_path, "Trained for {} Epochs, hexbin-fig {:.2f} loss.png".format(
                cur_epoch_nr, loss))
            , dpi=300)

    plt.show()
    plt.close()


def initialize_training():
    # Time-stopping start
    next_timeframe = [datetime.datetime.now()]

    df = dl.create_pfp_dataset()
    # or df = dl.read_out_other_fp_set(df, USE_FP)  # map4_fp or morgan4_fp from csv with additional already created FPs

    # wt%|mass fraction and mass concentration in g/mL are approximately the same for water
    df["poly_con"] = [b if isnan(a) else a for a, b in zip(df["polymer_concentration_wpercent"],
                                                          df["polymer_concentration_mass_conc"])]

    # Ion types from Salts will be saved positionally for the NN to learn with respective conc
    # handles (NH4)2 as two, H2PO4 and HPO4 as the same and H and OH as nothing
    salt_dic = {
        "NaCl": ["Na", "Cl"],
        "NaN3": ["Na", "N3"],
        "LiCl": ["Li", "Cl"],
        "CsCl": ["Cs", "Cl"],
        "RbCl": ["Rb", "Cl"],
        "KCl": ["K", "Cl"],
        "KI": ["K", "I"],
        "KBr": ["K", "Br"],
        "KF": ["K", "F"],
        "KOH": ["K"],
        "KSFO4": ["K", "SFO4"],
        "LiOH": ["Li"],
        "CsOH": ["Cs"],
        "RbOH": ["Rb"],
        "NaOH": ["Na"],
        "(NH4)2SO4": ["NH4", "NH4", "Cl"],
        "TBAAc": ["TBA", "Ac"],
        "SDS": ["Na", "dodecylsulfat"],
        "Na2HPO4": ["Na", "Na", "H2PO4"],
        "HCl": ["Cl"],
        "KH2PO4": ["K", "H2PO4"],
        "K2SO4": ["K", "K", "SO4"],
        "Na2SO4": ["Na", "Na", "SO4"],
        "Li2SO4": ["Li", "Li", "SO4"],
        "NaOAc": ["Na", "OAc"],
        "LiOAc": ["Li", "OAc"],
        "LiClO4": ["Li", "ClO4"],
        "LiI": ["Li", "I"],
        "NaI": ["Na", "I"],
        "NaClO4": ["Na", "ClO4"],
        "NaSCN": ["Na", "SCN"],
        "hexasodium calix[6]arenehexasulfonic acid": ["Na", "Na", "Na", "Na", "Na", "Na", "calix6"],
    }

    def ion_setter(salt_cols, salt_conc):  # salt concentration effect
        for column in salt_cols:
            for salt in column:
                if not salt in salt_dic.keys():
                    salt_dic[salt] = [salt]
                    # print("added {}".format(salt))
        del salt_dic[np.nan]
        ion_types = []
        for ions in salt_dic.values():
            for ion in ions:
                ion_types.append(ion) if ion not in ion_types else None
        print("{} Ion-types from {} Salt-Types".format(len(ion_types), len(salt_dic)))
        # print(ion_types)
        new_col_as_arr = []
        for columns in zip(*salt_cols, *salt_conc):
            temp_arr = np.zeros(len(ion_types))

            if not isinstance(columns[0], type(np.nan)):
                temp_ions = salt_dic[columns[0]]
                for ion in temp_ions:
                    temp_arr[ion_types.index(ion)] += columns[2]

            if not isinstance(columns[1], type(np.nan)):
                temp_ions = salt_dic[columns[1]]
                for ion in temp_ions:
                    temp_arr[ion_types.index(ion)] += columns[3]
            new_col_as_arr.append(temp_arr[1:])
        return new_col_as_arr

    salts_cols = df["additive1"], df["additive2"]
    salts_conc_cols = df["additive1_concentration_molar"], df["additive2_concentration_molar"]

    df["salts"] = ion_setter(salt_cols=salts_cols, salt_conc=salts_conc_cols)

    # using only one def_type
    # df = df[df["def_type"] == "A"].reset_index()

    def def_decision(def_type):
        aqu_point = int
        # this catches the NaN of type number
        if type(def_type) == type(0.1):
            aqu_point = "0.1"
        else:
            aqu_point = def_type.replace("DSC", "0.01").replace(
                "A", "0.1").replace("B", "0.2").replace("C", "0.5").replace(",", ".")
        return float(aqu_point)

    df["aqu_point"] = [def_decision(x) for x in df["def_type"]]
    df["pH"] = [7 if isnan(x) else x for x in df["pH"]]

    # concatenation of learning material
    df["togeth"] = df["Mn"].apply(
        lambda x: [(np.log10(x) / 6)]) + df["poly_con"].apply(
        lambda x: [x]) + df["fp"].apply(
        lambda x: x.tolist()) + df["aqu_point"].apply(
        lambda x: [x]) + df["pH"].apply(
        lambda x: [x]) + df["salts"].apply(
        lambda x: x.tolist())

    # cleaving out fragmentary entries
    global df_edit
    df_edit = df[["togeth", "Mn", "fp", "poly_con", "cloud_point"]].dropna()
    print(df_edit)

    Pm.Layer_size = len(df_edit["togeth"][0])

    # time-stopping breakpoint after dataset gen
    next_timeframe.append(datetime.datetime.now())

    print("preparing model...")
    model = TranTempPred()
    datamodule = DataManager(batch_size=Pm.BATCH_SIZE)
    trainer = Trainer(
        max_epochs=Pm.MAX_EPOCHS,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=Pm.MIN_DELTA, patience=Pm.EARLY_STOPPING_PATIENCE)])
    trainer.fit(model, datamodule)

    loss_deviation = model.loss
    print_plot(model, loss_deviation, save=True)

    # saving the model and all important to that after training
    df["pred"] = df["togeth"].apply(lambda x: float((model(torch.from_numpy(np.array(x)).float())).detach()))
    df["discrepancy"] = [pred - actual for pred, actual in zip(df["cloud_point"], df["pred"])]
    df.drop("togeth", axis=1).to_csv(os.path.join(Pm.export_path,
                                                  "{}+-{} of {} Epochs, {}FPs, {}Lr & {}Bs; {:.2f}loss.csv".format(
                                                      Pm.current_epoch_nr, Pm.EARLY_STOPPING_PATIENCE, Pm.MAX_EPOCHS,
                                                      Pm.FP_SIZE, Pm.LR, Pm.BATCH_SIZE, Pm.last_loss)),
                                     sep=";", decimal=",")

    print("saving model under {}".format(
        str(os.path.join(Pm.export_path)) + "complete_model_{}.pt".format(Pm.model_nr)))
    torch.save(model.state_dict(), os.path.join(Pm.export_path, "complete_model_{}.pt".format(Pm.model_nr)))

    # all we need to reuse/load the model later with the exact same dataset.
    dataset_of_model = [df_edit, Pm.Layer_size, Pm.current_epoch_nr]
    with open(os.path.join(Pm.export_path, "dataset_of_model_{}.pickle".format(Pm.model_nr)), "wb") as f:
        pickle.dump(dataset_of_model, f)

    # time-stopping finish
    next_timeframe.append(datetime.datetime.now())

    def time_to_readable(elapsed_time):
        e_mins, e_secs = divmod(elapsed_time.total_seconds(), 60)
        e_hours, e_mins = divmod(e_mins, 60)
        return e_hours, e_mins, e_secs

    elapsed_times = []
    for i in range(len(next_timeframe) - 1):
        elapsed_times.append(next_timeframe[i + 1] - next_timeframe[i])

    for elapsedTime in elapsed_times:
        print("this took {:.0f} hours, {:.0f} minutes and {:.0f} seconds".format(*time_to_readable(elapsedTime)))

    timing = ["{:.0f} h, {:.0f} m and {:.0f} s".format(*time_to_readable(e_time)) for e_time in elapsed_times]
    with open(os.path.join(Pm.export_path, "stats.txt"), mode="a") as f:
        f.write("Dataset generation and training took " + timing.__str__() + "\n")
        f.write(Pm.model_nr + " with following parameters: \n")
        f.write("Last MSE loss is {:.3f}".format(Pm.last_loss) + "\n")
        f.write("The initial layer size was %s " % Pm.Layer_size + "\n")
        f.write("LR: %s; BS: %s" % (Pm.LEARNING_RATE, Pm.BATCH_SIZE) + "\n")
        f.write("fp used: " + Pm.USE_FP + "with " + Pm.pfp_const_type + "\n")
        f.write("{} or up to {} less (Patience) of max {} "
                "Epochs with early stopping patience of and a min_delta of 0.01".format(
            Pm.current_epoch_nr, Pm.EARLY_STOPPING_PATIENCE, Pm.MAX_EPOCHS, Pm.EARLY_STOPPING_PATIENCE))


'''
    Caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    predtemp = model(
        torch.from_numpy(np.append(
            np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(Caffeine), fpSize=FP_SIZE), float), 0.1
        )).float())
    print("according to this model Caffeine would have an average Cp of {} :P".format(int(predtemp.detach().numpy())))
'''


def use_model(print_style="print_hexbin"):  # only works for datasets created with pfp!

    src_folder = dl.choose_source_folder()

    # reread pfp and dataset
    pickle_files_of_pfp = []
    dataset_files = []
    for files in os.listdir(src_folder):
        if files.endswith(".pickle"):
            if files.startswith("P_FP"):
                pickle_files_of_pfp.append(files)
            if files.startswith("dataset"):
                dataset_files.append(files)

    if len(pickle_files_of_pfp) != 1:
        raise Exception("No or more than one PFP pickle file found.")
    else:
        print("loading {}".format(*pickle_files_of_pfp))

    with open(os.path.join(src_folder, *dataset_files), "rb") as f:
        loaded_dataset = pickle.load(f)

    # handling whether loaded dataset already
    if len(loaded_dataset) > 3:
        Pm.current_epoch_nr = loaded_dataset[2]
    else:
        Pm.current_epoch_nr = "{} Epochs max".format(Pm.MAX_EPOCHS)

    global df_edit
    df_edit = loaded_dataset[0]

    '''
    When using other datasets for this model their input FP must reduced and combined with the rest for "togeth"
    load other fp and df, reduce fp with the fp-set loaded above and run following addition to combine:
    # df["togeth"] = df["Mn"].apply(
    #     lambda x: [(np.log10(x) / 6)]) + df["poly_con"].apply(
    #     lambda x: [x]) + df["fp"].apply(
    #     lambda x: x.tolist()) + df["aqu_point"].apply(
    #     lambda x: [x]) + df["pH"].apply(
    #     lambda x: [x]) + df["salts"].apply(
    #     lambda x: x.tolist())
    #
    # df_edit = df[["togeth", "Mn", "fp", "poly_con", "cloud_point"]].dropna() #cleanup
    '''

    Pm.Layer_size = len(df_edit["togeth"][0])
    # !this Layer_size will be set already when choosing which fp to use!
    print("source folder" + src_folder)
    loaded_model = TranTempPred()
    loaded_model.load_state_dict(torch.load(src_folder))
    loaded_model.eval()

    loss_deviation = loaded_model.loss

    def plot_single_sets():
        # redo splitting
        dataset_to_split = AddDataSet()
        split = (np.array([0.8, 0.1, 0.1]) * len(dataset_to_split)).astype(int).tolist()
        split[0] += len(dataset_to_split) - sum(split)
        train_set, val_set, test_set = random_split(dataset_to_split, split, generator=torch.Generator().manual_seed(7))
        sets = (train_set, val_set, test_set)
        set_names = ["train_set", "val_set", "test_set"]

        for set, set_name in zip(sets, set_names):
            set_x_data = torch.cat([torch.unsqueeze(x, 0) for (x, y) in set])
            set_y_data = torch.cat([torch.unsqueeze(y, 0) for (x, y) in set])
            y_hat = loaded_model(set_x_data)
            x = y_hat.detach().numpy()
            y = set_y_data.numpy()
            loss_function = loaded_model.loss
            loss = loss_function(y_hat, set_y_data)
            o_line = [[y.max(), y.min()], [y.max(), y.min()]]
            plt.hexbin(x, y, gridsize=30, cmap="ocean_r", linewidths=0.01)
            plt.plot(*o_line, color="orange")
            # plt.title(label="Training with {}/{} Epochs; loss is {:.2f}".format(Pm.current_epoch_nr, MAX_EPOCHS, loss))
            # calculating R^2/coefficient of determination:
            cod = 1 - (np.sum((y - x) ** 2)) / (np.sum((y - np.mean(y)) ** 2))
            plt.colorbar()
            plt.ylabel(r"Measured $T_{cp}$ [°C]")
            plt.xlabel(r"Predicted $T_{cp}$ [°C]")
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.savefig(fname=os.path.join(Pm.export_path, "{} MSE{:.2f}, CoD {:.2f}.png".format(set_name, loss, cod)),
                        dpi=300)
            plt.show()
            plt.close()

    if print_style == "single_sets":
        plot_single_sets()
    elif print_style == "detailed":
        print_plot(loaded_model, loss_deviation, save=True)
    else:
        print_plot_2(loaded_model, loss_deviation, Pm.current_epoch_nr, save=True)


if __name__ == "__main__":
    initialize_training()
    # use_model("single_sets")
