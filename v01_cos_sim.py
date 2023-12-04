# Encode dual-label molecular biomarker files with cosine sim loss VAE

# Call sig:
# conda activate py310tf210
# python3 v0.4c.py -f yyX/prot_X-files_v0.4b.tsv -v v0.4c -l proteomics -p prot -a _prot_ -r 1 -k -0.5 -m -1 -c 1
print('Begin setup, imports, and functions')
mdls_ttl = 'Cell line + CPTAC'
mdls = 'cptac_+_cell_line'

import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from scipy.spatial.distance import euclidean
from matplotlib.colors import ListedColormap
from joypy import joyplot
import time
import sys
plt.rcParams['text.usetex'] = True

strt_tm = time.time()

# Begin eval plots
def reduce(dta_typ_obj):
    reducer = umap.UMAP(n_components=2)
    scaled_data = StandardScaler().fit_transform(dta_typ_obj.iloc[:, 2:])
    embedding = reducer.fit_transform(scaled_data)
    emb_df = pd.DataFrame(embedding, index = dta_typ_obj.index)
    emb_lbld = pd.concat([emb_df, dta_typ_obj[['cancer_type', 'model_type']]], axis = 1)
    emb_lbld.columns = ['UMAP_1', 'UMAP_2', 'cancer_type', 'model_type']
    return emb_lbld

def umap_plot_to_disk(emb_lbld, mdls_ttl, dta_ttl, dta_typ, mdls, epochs, latent_dim):
    color_toggle = 'orng'
    cancer_type_abbreviation_mapping = {
        'carcinoma': 'Custom_Darker_Orange',
        'not_carcinoma': 'Custom_Orange',}
    tcga_colors = pd.read_csv('plot_color_files/'+color_toggle+'_colors.tsv', sep = '\t', index_col = 0)
    unique_cancer_types = emb_lbld['cancer_type'].unique()
    custom_palette = {cancer_type: tcga_colors.loc[cancer_type_abbreviation_mapping.get(
        cancer_type, 'Unknown'), 'cohort_color'] for cancer_type in unique_cancer_types}
    emb_lbld.columns = ['UMAP_1', 'UMAP_2', 'Cancer type', 'Model type']
    plt.figure(figsize=(5, 5))
    marker_dict = {'Tumor': '^', 'cell line': 'o'}
    sns.scatterplot(data=emb_lbld, x='UMAP_1', y='UMAP_2',
                    hue='Cancer type', style='Model type', markers=marker_dict,
                    palette=custom_palette, legend='full',
                    s = 200)
    plt.xlabel('UMAP_1', fontsize=16)
    plt.ylabel('UMAP_2', fontsize=16)
    plt.legend(title='Cancer Type', loc='upper left', bbox_to_anchor=(1, 1))
    plt.suptitle(mdls_ttl +', '+ dta_ttl, y = 1.002, fontsize = 20)
    plt.title('n = '+str(len(emb_lbld))+encdg_stts_ttl+', epochs = '+str(epochs)+', latent dim = '+str(latent_dim), 
              fontsize = 18)
    legend = plt.legend(title='Sample attributes', title_fontsize='14', loc='upper left',
                        bbox_to_anchor=(1, 1), fontsize=12)
    headers_to_bold = ['Cancer type', 'Model type']
    for text in legend.texts:
        if text.get_text() in headers_to_bold:
            text.set_weight('bold')
    plt.rcParams['text.usetex'] = True
    plt.savefig(Path(log_dir, 'umap_'+dta_typ+'_'+mdls+'_'+encdg_stts_nam+'.png'),
                bbox_inches = 'tight', dpi = 300)
    return 'UMAP written to disk'

# LogReg for both greyscaled and colored bar, mode is model type or cancer type
def log_reg(dta_typ_obj, mode):
    col_X_strt = 2
    f1_stor_frm = pd.DataFrame()
    for i in list(range(0, 10)):
        trn = dta_typ_obj.sample(round(len(dta_typ_obj) * .8))
        tst = dta_typ_obj.loc[~dta_typ_obj.index.isin(trn.index)]
        X_trn = trn.iloc[:, col_X_strt:]
        X_tst = tst.iloc[:, col_X_strt:]
        y_trn = trn[mode]
        y_tst = tst[mode]
        clf = LogisticRegression(max_iter=1000).fit(X_trn, y_trn)
        y_pred = clf.predict(X_tst)
        f1_by_class = f1_score(y_tst, y_pred, average=None)
        f1_df = pd.DataFrame({'Label': list(y_tst.unique()),
                              'F1_Score': f1_by_class})
        f1_stor_frm = pd.concat([f1_stor_frm, f1_df], axis = 0)
    return f1_stor_frm

def logreg_model_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl):
    plt.figure(figsize=(8, 4.5))
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)
    sns.barplot(x='Label', y='F1_Score', data=f1_stor_frm, palette=['#666666', '#999999'],
               errorbar=None)
    sns.swarmplot(x='Label', y='F1_Score', data=f1_stor_frm, color='#333333', size=10)
    plt.suptitle('Logistic regression, '+mode_ttl+', '+dta_ttl,
                 fontsize=24, y = 1.03)
    plt.title(mdls_ttl+encdg_stts_ttl+', epochs = '+str(epochs)+', latent dim = '+str(latent_dim), fontsize=20)
    plt.xlabel('Model Type', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)
    new_labels = [f"{label}, n = {sample_counts[label]}" for label in sample_counts.keys()]
    plt.xticks(ticks=range(len(new_labels)), labels=new_labels, fontsize=20)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.rcParams['text.usetex'] = True #
    plt.savefig(Path(log_dir, 'logreg_'+mode+'_'+dta_typ+'_'+mdls+'_'+encdg_stts_nam+'.png'),
                bbox_inches='tight', dpi = 300)

def logreg_cancer_plot(f1_stor_frm, mdls, data_name, latent_dim, epochs, mode, mode_ttl): # chk
    color_toggle = 'orng'
    tcga_colors = pd.read_csv('plot_color_files/'+color_toggle+'_colors.tsv',
                          sep = '\t')
    tcga_color_mapping = dict(zip(tcga_colors['tcga_cohorts'], tcga_colors['cohort_color']))
    unique_labels = f1_stor_frm['Label'].unique()
    palette_dict = {}
    cancer_type_abbreviation_mapping = {
        'carcinoma': 'Custom_Darker_Orange',
        'not_carcinoma': 'Custom_Orange',}
    for label in unique_labels:
        tcga_abbreviation = cancer_type_abbreviation_mapping.get(label)
        color = tcga_color_mapping.get(tcga_abbreviation)
        if color:
            palette_dict[label] = color
    plt.figure(figsize=(8, 4))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        x='Label', y='F1_Score', data=f1_stor_frm,
        palette=palette_dict,
        errorbar=None)
    sns.swarmplot(x='Label', y='F1_Score', data=f1_stor_frm,
                  color='#333333', size=7)
    plt.suptitle('Logistic regression, '+mode_ttl+', '+dta_ttl,
             fontsize=24, y = 1.04)
    plt.title(mdls_ttl+encdg_stts_ttl+', epochs = '+str(epochs)+', latent dim = '+str(latent_dim), fontsize=20)
    plt.xlabel('Cancer type', fontsize=20)
    plt.ylabel('F1 Score', fontsize=20)
    new_labels = [f"{label}, n = {sample_counts[label]}" for label in sample_counts.keys()]
    plt.xticks(ticks=range(len(new_labels)), labels=new_labels, fontsize=20)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.rcParams['text.usetex'] = True
    plt.savefig(Path(log_dir, 'logreg_'+mode+'_'+dta_typ+'_'+mdls+'_'+encdg_stts_nam+'.png'),
                bbox_inches='tight', dpi = 300)

# Euclicean distance, model type
def mdl_typ_dist(sample, features, df):
    other_types = df[df['model_type'] != sample['model_type']]
    mean_features_other_types = other_types[features].mean()
    distance = euclidean(sample[features], mean_features_other_types)
    return distance

# Euclidean distance, cancer type
def cncr_typ_dist(sample, features, df):
    other_types = df[df['cancer_type'] != sample['cancer_type']]
    mean_features_other_types = other_types[features].mean()
    distance = euclidean(sample[features], mean_features_other_types)
    return distance

def euc_dstncs(dta_typ_obj): # 173
    dta_typ_copy = dta_typ_obj.copy()
    feature_columns = dta_typ_copy.columns[2:]
    dta_typ_copy['mdl_typ_dstncs'] = dta_typ_copy.apply(
        lambda row: mdl_typ_dist(row, feature_columns, dta_typ_copy), axis=1)
    dta_typ_copy['cncr_typ_dstncs'] = dta_typ_copy.apply(
        lambda row: cncr_typ_dist(row, feature_columns, dta_typ_copy), axis=1)
    new_cols = ['cancer_type', 'model_type', 'cncr_typ_dstncs', 'mdl_typ_dstncs']
    dta_typ_copy = dta_typ_copy[new_cols]
    return dta_typ_copy

def euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances):
    plt.figure()
    joyplot(data=sorted_df[[mode, dstnc_typ]], by=mode,
        figsize=(10, 6.5), colormap=custom_colormap,
        fade=True)
    for y_value, cancer_type in zip(y_values, sorted_df[mode].unique()):
        count = dict(sorted_df[mode].value_counts())[cancer_type]
        x_position = sorted_df[dstnc_typ].max()
        plt.annotate(f"n={count}", xy=(x_position, y_value), verticalalignment='center')
    plt.suptitle('Euclidean Distances, '+mode_ttl+', '+dta_ttl,
                 fontsize=30, y = 1.01)
    plt.title(mdls_ttl+ ', n = '+str(sorted_df.shape[0])+encdg_stts_ttl,
              y = .92, x = .42, fontsize = 26)
    plt.rcParams['text.usetex'] = True
    plt.annotate(
        r'Variance of means: $\mathbf{' + f'{average_distances.var():.3f}' + '}$',
        xy=(0.15, 0.87), xycoords='axes fraction',
        ha='right', va='top')
    plt.savefig(Path(log_dir, 'eucdist_'+mode+'_'+dta_typ+'_'+mdls+'_'+encdg_stts_nam+'.png'),
        bbox_inches='tight', dpi = 300)

def ave_dist_cncr(dta_typ_obj, mode_ttl, mode, dstnc_typ):
    average_distances = dta_typ_obj.groupby(mode)[dstnc_typ].mean().sort_values(ascending=False)
    sorted_df = dta_typ_obj.loc[dta_typ_obj[mode].isin(average_distances.index)]
    sorted_df[mode] = pd.Categorical(
        sorted_df[mode], categories=average_distances.index, ordered=True)
    sorted_df = sorted_df.sort_values(mode)
    color_toggle = 'orng'
    tcga_colors = pd.read_csv('plot_color_files/'+color_toggle+'_colors.tsv', sep = '\t', index_col = 0)
    cancer_type_abbreviation_mapping = {
        'carcinoma': 'Custom_Darker_Orange',
        'not_carcinoma': 'Custom_Orange',}
    custom_color_list = [tcga_colors.loc[cancer_type_abbreviation_mapping[cancer_type],'cohort_color'] for cancer_type in average_distances.index]
    custom_colormap = ListedColormap(custom_color_list)
    return sorted_df, custom_colormap, average_distances

def ave_dist_mdl(dta_typ_obj, mode_ttl, mode, dstnc_typ):
    average_distances = dta_typ_obj.groupby(mode)[dstnc_typ].mean().sort_values(ascending=False)
    sorted_df = dta_typ_obj.loc[dta_typ_obj[mode].isin(average_distances.index)]
    sorted_df[mode] = pd.Categorical(
        sorted_df[mode], categories=average_distances.index, ordered=True)
    sorted_df = sorted_df.sort_values(mode)
    color_toggle = 'grey'
    abbreviation_mapping = {
    'cell line': 'cell line',
    'Tumor': 'Tumor',
    'HCMI': 'HCMI', # devel
                }
    grey_colors = pd.read_csv('plot_color_files/'+color_toggle+'_scale.tsv', sep = '\t', index_col = 0)
    custom_color_list = [grey_colors.loc[
                     abbreviation_mapping[
                     model_type],'quant_mode_color'] for model_type in average_distances.index]
    custom_colormap = ListedColormap(custom_color_list)
    return sorted_df, custom_colormap, average_distances

# Begin VAE
model_type_encoder = LabelEncoder()
cancer_type_encoder = LabelEncoder()

def cos_sim_modl(df: pd.DataFrame, model_type_column='model_type') -> (dict, dict):
    df_cell_line = df[df[model_type_column] == 'cell line'].drop(columns=[model_type_column])
    df_tumor = df[df[model_type_column] == 'Tumor'].drop(columns=[model_type_column])
    cosine_similarities_cell_line = {}
    cosine_similarities_tumor = {}
    valid_columns = df_cell_line.select_dtypes(include=[np.number]).columns
    mean_vector_cell_line = df_cell_line[valid_columns].mean(axis=0).values
    mean_vector_tumor = df_tumor[valid_columns].mean(axis=0).values
    
    for index, row in df_cell_line.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cell_line = distance.cosine(sample_vector, mean_vector_cell_line)
        sim_to_tumor = distance.cosine(sample_vector, mean_vector_tumor)
        cosine_similarities_cell_line[index] = (sim_to_cell_line, sim_to_tumor)

    for index, row in df_tumor.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cell_line = distance.cosine(sample_vector, mean_vector_cell_line)
        sim_to_tumor = distance.cosine(sample_vector, mean_vector_tumor)
        cosine_similarities_tumor[index] = (sim_to_cell_line, sim_to_tumor)
    cosine_similarities_tumor.update(cosine_similarities_cell_line)
    cosine_similarities = cosine_similarities_tumor

    intra_cluster_tensor = list(range(df.shape[0]))
    inter_cluster_tensor = list(range(df.shape[0]))

    assert len(intra_cluster_tensor) == len(df), "Length of list is not as expected"
    assert len(inter_cluster_tensor) == len(df), "Length of list is not as expected"

    for key in cosine_similarities.keys():
        intra_cluster_tensor[key] = cosine_similarities[key][0]
        inter_cluster_tensor[key] = cosine_similarities[key][1]

    intra_cluster_tensor = tf.convert_to_tensor(intra_cluster_tensor, dtype=np.float32)
    inter_cluster_tensor = tf.convert_to_tensor(inter_cluster_tensor, dtype=np.float32)

    return intra_cluster_tensor, inter_cluster_tensor

def cos_sim_cncr(df: pd.DataFrame, cancer_type_column='cancer_type') -> (dict, dict):
    df_cncr_a = df[df[cancer_type_column] == 'carcinoma'].drop(columns=[cancer_type_column]) # Hard code
    df_cncr_b = df[df[cancer_type_column] == 'not_carcinoma'].drop(columns=[cancer_type_column])
    cosine_similarities_cncr_a = {}
    cosine_similarities_cncr_b = {}
    valid_columns = df_cncr_a.select_dtypes(include=[np.number]).columns
    mean_vector_cncr_a = df_cncr_a[valid_columns].mean(axis=0).values
    mean_vector_cncr_b = df_cncr_b[valid_columns].mean(axis=0).values
    
    for index, row in df_cncr_a.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cncr_a = distance.cosine(sample_vector, mean_vector_cncr_a)
        sim_to_cncr_b = distance.cosine(sample_vector, mean_vector_cncr_b)
        cosine_similarities_cncr_a[index] = (sim_to_cncr_a, sim_to_cncr_b)

    for index, row in df_cncr_b.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cncr_a = distance.cosine(sample_vector, mean_vector_cncr_a)
        sim_to_cncr_b = distance.cosine(sample_vector, mean_vector_cncr_b)
        cosine_similarities_cncr_b[index] = (sim_to_cncr_a, sim_to_cncr_b)
    cosine_similarities_cncr_b.update(cosine_similarities_cncr_a)
    cosine_similarities = cosine_similarities_cncr_b

    intra_cluster_tensor = list(range(df.shape[0]))
    inter_cluster_tensor = list(range(df.shape[0]))

    assert len(intra_cluster_tensor) == len(df), "Length of list is not as expected"
    assert len(inter_cluster_tensor) == len(df), "Length of list is not as expected"

    for key in cosine_similarities.keys():
        intra_cluster_tensor[key] = cosine_similarities[key][0]
        inter_cluster_tensor[key] = cosine_similarities[key][1]

    intra_cluster_tensor = tf.convert_to_tensor(intra_cluster_tensor, dtype=np.float32)
    inter_cluster_tensor = tf.convert_to_tensor(inter_cluster_tensor, dtype=np.float32)

    return intra_cluster_tensor, inter_cluster_tensor

# Define the Sampling Layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the VAE class
class VAE(keras.Model):
    def __init__(self, encoder, decoder, columns, **kwargs):
        super().__init__(**kwargs)
        self.encoder: keras.Model = encoder
        self.decoder: keras.Model = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.modl_dstnc_loss_tracker = keras.metrics.Mean(name="modl_dstnc_loss")
        self.cncr_dstnc_loss_tracker = keras.metrics.Mean(name="cncr_dstnc_loss")
        self.columns = columns

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.modl_dstnc_loss_tracker,
            self.cncr_dstnc_loss_tracker,
        ] 

    def train_step(self, data):
        with tf.GradientTape() as tape:
            converted_data: pd.DataFrame = pd.DataFrame(data.numpy(), columns=self.columns)
            
            model_type = converted_data["model_type"]
            model_type = model_type.astype(int)
            model_type = model_type_encoder.inverse_transform(model_type)
            data = converted_data.drop(columns=["model_type"])
            assert "model_type" not in data.columns, "model_type should not be in data"

            cancer_type = converted_data["cancer_type"]
            cancer_type = cancer_type.astype(int)
            cancer_type = cancer_type_encoder.inverse_transform(cancer_type)
            data = data.drop(columns=["cancer_type"])
            assert "cancer_type" not in data.columns, "cancer_type should not be in data"

            data = tf.convert_to_tensor(data)

            z_mean, z_log_var, z = self.encoder(data)
            
            modl_labeled_embeddings: pd.DataFrame = pd.DataFrame(z.numpy())
            modl_labeled_embeddings["model_type"] = model_type              

            cncr_labeled_embeddings: pd.DataFrame = pd.DataFrame(z.numpy()) 
            cncr_labeled_embeddings["cancer_type"] = cancer_type            

            modl_ntra_clstr_dstnc, modl_nter_clstr_dstnc = cos_sim_modl(
                df=modl_labeled_embeddings,
                model_type_column='model_type')

            cncr_ntra_clstr_dstnc, cncr_nter_clstr_dstnc = cos_sim_cncr(
                df=cncr_labeled_embeddings,
                cancer_type_column='cancer_type')

            # 4 coefficients - pass on commanc line
            # recon_coef = 1
            # kl_coef = - 0.5
            # mdl_coef = -1
            # cncr_coef = -1

            reconstruction = self.decoder(z)
            reconstruction_loss = recon_coef * data.shape[1] * keras.losses.binary_crossentropy(data, reconstruction)
            kl_loss = kl_coef * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

            modl_distance_loss = mdl_coef * modl_ntra_clstr_dstnc
            cncr_distance_loss = cncr_coef * cncr_ntra_clstr_dstnc
            total_loss = reconstruction_loss + kl_loss + modl_distance_loss + cncr_distance_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.modl_dstnc_loss_tracker.update_state(modl_distance_loss)
        self.cncr_dstnc_loss_tracker.update_state(cncr_distance_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "modl_dstnc_loss": self.modl_dstnc_loss_tracker.result(),
            "cncr_dstnc_loss": self.cncr_dstnc_loss_tracker.result(),
        }

# Build Encoder
def build_encoder(feature_dim, latent_dim) -> keras.Model:
    encoder_inputs = keras.Input(shape=(feature_dim,), name="input_1")
    x = keras.layers.Dense(latent_dim, kernel_initializer='glorot_uniform', name="encoder_dense_1")(encoder_inputs)
    x = keras.layers.BatchNormalization(name="batchnorm")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Build Decoder
def build_decoder(feature_dim, latent_dim) -> keras.Model:
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(feature_dim, kernel_initializer='glorot_uniform', activation='sigmoid')(latent_inputs)
    decoder_outputs = x
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--file", action="store", type=str, required=True)
    parser.add_argument("-v", "--version", action="store", type=str, required=True)
    parser.add_argument("-l", "--dta_ttl", action="store", type=str, required=True)
    parser.add_argument("-p", "--dta_typ", action="store", type=str, required=True)
    parser.add_argument("-a", "--log_dta", action="store", type=str, required=True)
    parser.add_argument("-r", "--recon_coef", action="store", type=int, required=True)
    parser.add_argument("-k", "--kl_coef", action="store", type=float, required=True)
    parser.add_argument("-m", "--mdl_coef", action="store", type=float, required=True)
    parser.add_argument("-c", "--cncr_coef_coef", action="store", type=float, required=True)    
    
    args = parser.parse_args()

    v: str = args.version
    file: str = args.file
    dta_ttl: str = args.dta_ttl
    dta_typ: str = args.dta_typ
    log_dta: str = args.log_dta
    recon_coef: int = args.recon_coef
    kl_coef: float = args.kl_coef
    mdl_coef: float = args.mdl_coef
    cncr_coef: float = args.cncr_coef_coef

    log_dir = Path("logs", v + log_dta + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +
              '_r'+str(recon_coef)+'_k'+str(kl_coef)+'_m'+str(mdl_coef)+'_c'+str(cncr_coef))
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Read data, file from command line arg path
    train_df = pd.read_csv(file,
                   sep = '\t', index_col = 0)
    print(' ')
    print('train_df shape:', train_df.shape)
    print(' ')
    selected_df = train_df.iloc[:, 2:]
    
    feature_count = selected_df.shape[1]

    # Take-off point 1, train_df plot lables- variables are reset after VAE for encdd and decdd
    epochs = 0
    latent_dim = 'NA'
    
    encdg_stts_ttl = ', train_df' # var to end of all plot titles
    encdg_stts_nam = 'train_df'   # plot file name on disk
    
    # UMAP
    reduced = reduce(train_df)
    umap_plot_to_disk(reduced, mdls_ttl, dta_ttl, dta_typ, mdls, epochs, latent_dim)

    print('start train_df quantifications')
    # Create distance frame
    euc_df = euc_dstncs(train_df)

    # LogReg and EucDist on model type
    mode_ttl = 'model type'
    mode = 'model_type'

    f1_stor_frm = log_reg(train_df, mode)
    sample_counts = dict(train_df[mode].value_counts())
    logreg_model_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)
    
    y_values = np.linspace(0.6, 0.15, 2)
    dstnc_typ = 'mdl_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_mdl(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)

    # LogReg and EucDist on cancer type
    mode_ttl = 'cancer type'
    mode = 'cancer_type'
    
    f1_stor_frm = log_reg(train_df, mode)
    sample_counts = dict(train_df[mode].value_counts())
    logreg_cancer_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)    
    
    y_values = np.linspace(0.6, 0.15, 2) # for multi-categorical modification
    dstnc_typ = 'cncr_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_cncr(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)

    # Begin VAE encodding
    latent_dim = 50 # overwrite from train_df plots
    learning_rate = 0.001
    epochs = 10     # overwrite from train_df plots
    batch_size = 72
    
    scaler = MinMaxScaler()
    selected_df = pd.DataFrame(
        scaler.fit_transform(selected_df),
        columns=selected_df.columns,
        index=selected_df.index)
    
    selected_df["model_type"] = train_df["model_type"]
    selected_df["cancer_type"] = train_df["cancer_type"]

    # Plot take-off point for scaled data - placehold 
    
    selected_df["model_type"] = model_type_encoder.fit_transform(selected_df["model_type"])
    selected_df["model_type"] = selected_df["model_type"].astype(int)
    assert selected_df["model_type"].nunique() == 2, "There should be two classes"
    
    selected_df["cancer_type"] = cancer_type_encoder.fit_transform(selected_df["cancer_type"])
    selected_df["cancer_type"] = selected_df["cancer_type"].astype(int)
    assert selected_df["cancer_type"].nunique() == 2, "There should be two classes"
    
    print(selected_df.shape)
    
    encoder = build_encoder(feature_count, latent_dim)  # feat count set above, lat dim is a var
    decoder = build_decoder(feature_count, latent_dim)
    vae = VAE(encoder, decoder, columns=selected_df.columns)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), run_eagerly=True)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Path(log_dir, "weights"),
                                                     save_weights_only=True,
                                                     verbose=1)
    
    history = vae.fit(selected_df, epochs=epochs, batch_size=batch_size, shuffle=True,
                      callbacks=[tensorboard_callback, cp_callback])
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(log_dir, "history.tsv"), sep='\t', index=False)
    vae.encoder.save(Path(log_dir, "encoder"))
    vae.decoder.save(Path(log_dir, "decoder"))

    # Loss plot
    plot_df = history_df
    xlab = 'Epoch'
    axis_font_size = 18
    
    fig, main_ax = plt.subplots(figsize=(10, 8))
    main_ax.grid(False)
    main_ax.plot(plot_df['loss'], label='Total Loss', color='blue')
    main_ax.set_title('Overall Loss and Individual Loss Components, ' + dta_ttl, fontsize = 20)
    main_ax.set_xlabel(xlab, fontsize = axis_font_size)
    main_ax.set_ylabel('Total Loss', fontsize = axis_font_size)
    main_ax.legend()
    
    width, height = .25, .25
    a, b = .3, .65
    inset_title_font_size = 14
    
    # Reconstruction loss, upper left
    ax1 = main_ax.inset_axes([a, b, width, height])  # x, y, width, height
    ax1.grid(False) #format check
    ax1.plot(plot_df['reconstruction_loss'], label='Recon Loss', color='green')
    ax1.set_title('Reconstruction Loss', fontsize = inset_title_font_size)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel('Recon Loss')
    
    # KL loss, upper right
    ax2 = main_ax.inset_axes([b, b, width, height])
    
    ax2.plot(plot_df['kl_loss'], label='KL Loss', color='red')
    ax2.set_title('KL Loss', fontsize = inset_title_font_size)
    ax2.set_xlabel(xlab)
    ax2.set_ylabel('KL Loss')
    
    # Distance loss, lower left
    ax3 = main_ax.inset_axes([a, a, width, height])
    
    ax3.plot(plot_df['modl_dstnc_loss'], label='Model Distance Loss', color='orange')
    ax3.set_title('Model Distance Loss', fontsize = inset_title_font_size)
    ax3.set_xlabel(xlab)
    ax3.set_ylabel('Model Distance Loss')
    
    # Distance loss, lower right
    ax3 = main_ax.inset_axes([b, a, width, height])
    
    ax3.plot(plot_df['cncr_dstnc_loss'], label='Cancer Distance Loss', color='yellow')
    ax3.set_title('Cancer Distance Loss', fontsize = inset_title_font_size)
    ax3.set_xlabel(xlab)
    ax3.set_ylabel('Cancer Distance Loss')
    
    plt.tight_layout()
    plt.savefig(Path(log_dir, 'loss_'+dta_typ+'_'+mdls+'_'+encdg_stts_nam+'.png'),
                bbox_inches = 'tight', dpi = 300)

# Encodded evals
    # Load trained encoder
    encoder = tf.keras.models.load_model(Path(log_dir, "encoder"))
    # predict on train data
    dta_typ_obj = pd.DataFrame(encoder.predict(train_df.iloc[:, 2:])[0])
    # Put index, header, and labels on latent object
    dta_typ_obj.index = train_df.index
    str_cols = [dta_typ + '_' + str(chi) for chi in dta_typ_obj.columns]
    dta_typ_obj.columns = str_cols
    dta_typ_obj = pd.concat([train_df[['cancer_type', 'model_type']], dta_typ_obj], axis = 1)
    
    encdg_stts_ttl = ', encdd'
    encdg_stts_nam = 'encdd'
    print('start encoded quantifications')
    
    # UMAP
    reduced = reduce(dta_typ_obj)
    umap_plot_to_disk(reduced, mdls_ttl, dta_ttl, dta_typ, mdls, epochs, latent_dim)

    euc_df = euc_dstncs(dta_typ_obj)

    # LogReg and EucDist on model type
    mode_ttl = 'model type'
    mode = 'model_type'
    
    y_values = np.linspace(0.6, 0.15, 2)
    f1_stor_frm = log_reg(train_df, mode)
    sample_counts = dict(train_df[mode].value_counts())
    logreg_model_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)

    dstnc_typ = 'mdl_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_mdl(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)

    # LogReg and EucDist on cancer type
    mode_ttl = 'cancer type'
    mode = 'cancer_type'
    
    y_values = np.linspace(0.6, 0.15, 2)
    f1_stor_frm = log_reg(train_df, mode)
    sample_counts = dict(train_df[mode].value_counts())
    logreg_cancer_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)

    dstnc_typ = 'cncr_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_cncr(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)

# Decoded evaluations
    decoder = tf.keras.models.load_model(Path(log_dir, "decoder"))
    dta_typ_obj = pd.DataFrame(decoder.predict(encoder.predict(train_df.iloc[:, 2:])[0]))
    dta_typ_obj.index = train_df.index
    str_cols = [dta_typ + '_' + str(chi) for chi in dta_typ_obj.columns]
    dta_typ_obj.columns = str_cols
    dta_typ_obj = pd.concat([train_df[['cancer_type', 'model_type']], dta_typ_obj], axis = 1)
    
    encdg_stts_ttl = ', decdd'
    encdg_stts_nam = 'decdd'
    print('start decoded quantifications')
    
    # UMAP
    reduced = reduce(dta_typ_obj)
    umap_plot_to_disk(reduced, mdls_ttl, dta_ttl, dta_typ, mdls, epochs, latent_dim)
    
    euc_df = euc_dstncs(dta_typ_obj)
    
    # LogReg and EucDist on model type
    mode_ttl = 'model type'
    mode = 'model_type'
    
    y_values = np.linspace(0.6, 0.15, 2)
    f1_stor_frm = log_reg(dta_typ_obj, mode)
    sample_counts = dict(dta_typ_obj[mode].value_counts())
    logreg_model_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)
    
    dstnc_typ = 'mdl_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_mdl(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)
    
    # LogReg and EucDist on cancer type
    mode_ttl = 'cancer type'
    mode = 'cancer_type'
    
    y_values = np.linspace(0.6, 0.15, 2)
    f1_stor_frm = log_reg(dta_typ_obj, mode)
    sample_counts = dict(dta_typ_obj[mode].value_counts())
    logreg_cancer_plot(f1_stor_frm, mdls, dta_typ, latent_dim, epochs, mode, mode_ttl)
    
    dstnc_typ = 'cncr_typ_dstncs'
    sorted_df, custom_colormap, average_distances = ave_dist_cncr(euc_df, mode_ttl, mode, dstnc_typ)
    euc_plot(y_values, sorted_df, custom_colormap, dstnc_typ, average_distances)

    print(f"Wall time: {time.time() - strt_tm:.1f} seconds")