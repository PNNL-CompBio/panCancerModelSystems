# Version 0, programatically functional composite loss VAE
# Cancer model system feature distribution distance loss term integrated
# $ conda activate python3.10tensorflow2.10
# $ python3 vae_v0.py -f data/<model_systems>_<data_type>.tsv

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial import distance  # for cosine similarity distance
import numpy as np
from sklearn.preprocessing import LabelEncoder

log_dir = Path("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not log_dir.exists():
    log_dir.mkdir(parents=True, exist_ok=True)

model_type_encoder = LabelEncoder()
cancer_type_encoder = LabelEncoder()


def calculate_cosine_similarity(df: pd.DataFrame, model_type_column='model_type') -> (dict, dict):
    # Separate the DataFrame into two based on the model_type
    df_cell_line = df[df[model_type_column] == 'cell line'].drop(columns=[model_type_column])
    df_tumor = df[df[model_type_column] == 'Tumor'].drop(columns=[model_type_column])

    # Initialize dictionaries to store the results
    cosine_similarities_cell_line = {}
    cosine_similarities_tumor = {}

    # Identify the columns that are common and valid for mean calculation
    valid_columns = df_cell_line.select_dtypes(include=[np.number]).columns

    # Calculate the mean vector for each class for the relevant columns
    mean_vector_cell_line = df_cell_line[valid_columns].mean(axis=0).values
    mean_vector_tumor = df_tumor[valid_columns].mean(axis=0).values

    # Loop through each sample in cell_line and calculate cosine similarity to both mean vectors
    for index, row in df_cell_line.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cell_line = distance.cosine(sample_vector, mean_vector_cell_line)
        sim_to_tumor = distance.cosine(sample_vector, mean_vector_tumor)
        cosine_similarities_cell_line[index] = (sim_to_cell_line, sim_to_tumor)

    # Loop through each sample in tumor and calculate cosine similarity to both mean vectors
    for index, row in df_tumor.iterrows():
        sample_vector = row[valid_columns].values
        sim_to_cell_line = distance.cosine(sample_vector, mean_vector_cell_line)
        sim_to_tumor = distance.cosine(sample_vector, mean_vector_tumor)
        cosine_similarities_tumor[index] = (sim_to_cell_line, sim_to_tumor)

    # Combine the 2 dist dicts into one dict
    cosine_similarities_tumor.update(cosine_similarities_cell_line)
    cosine_similarities = cosine_similarities_tumor

    # Prefilled 0s lists, number of dictionary keys
    # Modified for truncated batch size (final batch of epoch)
    intra_cluster_tensor = list(range(df.shape[0]))
    inter_cluster_tensor = list(range(df.shape[0]))

    assert len(intra_cluster_tensor) == len(df), "Length of list is not as expected"
    assert len(inter_cluster_tensor) == len(df), "Length of list is not as expected"

    # Populate the zero lists with distance scores
    for key in cosine_similarities.keys():
        intra_cluster_tensor[key] = cosine_similarities[key][0]
        inter_cluster_tensor[key] = cosine_similarities[key][1]

    # Convert lists to tensors
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
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.distance_loss_tracker = keras.metrics.Mean(name="distance_loss")

        self.columns = columns  # Label vector to cosine similarity, dropped prior to .fit()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.distance_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            converted_data: pd.DataFrame = pd.DataFrame(data.numpy(), columns=self.columns)
            model_type = converted_data["model_type"]
            model_type = model_type.astype(int)
            model_type = model_type_encoder.inverse_transform(model_type)
            data = converted_data.drop(columns=["model_type"])  # Drop extracted cancer model system labels
            assert "model_type" not in data.columns, "model_type should not be in data"

            cancer_type = converted_data["cancer_type"]
            cancer_type = cancer_type.astype(int)
            cancer_type = cancer_type_encoder.inverse_transform(cancer_type)
            data = data.drop(columns=["cancer_type"]) # Drop extracted cancer type labels
            assert "cancer_type" not in data.columns, "cancer_type should not be in data"

            data = tf.convert_to_tensor(data)

            z_mean, z_log_var, z = self.encoder(data)

            labeled_embeddings: pd.DataFrame = pd.DataFrame(z.numpy())
            labeled_embeddings["model_type"] = model_type
            # print('\n\n\n')
            # print("Labeled Embeddings")
            # print(labeled_embeddings)
            # input() # "return break"

            intra_cluster_distance, inter_cluster_distance = calculate_cosine_similarity(
                df=labeled_embeddings,
                model_type_column='model_type')

            # print("Distances:")
            # print("Intra Cluster:")
            # print(intra_cluster_distance)
            # print("Inter Cluster:")
            # print(inter_cluster_distance)
            # input()

            reconstruction = self.decoder(z)

            # Use integers / floats as coefficients, sign flips for tuning cacer model system platform correction
            reconstruction_loss = .1 * data.shape[1] * keras.losses.binary_crossentropy(data, reconstruction)
            kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            distance_loss = - 100 * inter_cluster_distance

            total_loss = reconstruction_loss + kl_loss + distance_loss

            # print("Rec Loss")
            # print(reconstruction_loss)
            # print("kl Loss")
            # print(kl_loss)
            # print("Distance loss")
            # print(distance_loss)
            # print("Total loss")
            # print(total_loss)
            # input()

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.distance_loss_tracker.update_state(distance_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "distance_loss": self.distance_loss_tracker.result(),
        }


# Build Encoder
def build_encoder(feature_dim, latent_dim):
    encoder_inputs = keras.Input(shape=(feature_dim,), name="input_1")
    x = keras.layers.Dense(latent_dim, kernel_initializer='glorot_uniform', name="encoder_dense_1")(encoder_inputs)
    x = keras.layers.BatchNormalization(name="batchnorm")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# Build Decoder
def build_decoder(feature_dim, latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(feature_dim, kernel_initializer='glorot_uniform', activation='sigmoid')(latent_inputs)
    decoder_outputs = x
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


# Parameters
latent_dim = 50
learning_rate = 0.001
epochs = 30
batch_size = 128

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", action="store", type=str, required=True)

    args = parser.parse_args()

    file: str = args.file

    df = pd.read_csv(file, sep='\t', index_col=0)

    selected_df = df.iloc[:, 2:]

    feature_count = selected_df.shape[1] # to encoder, decoder build

    scaler = MinMaxScaler()
    selected_df = pd.DataFrame(
        scaler.fit_transform(selected_df),
        columns=selected_df.columns,
        index=selected_df.index)

    selected_df["model_type"] = df["model_type"]
    selected_df["cancer_type"] = df["cancer_type"]

    selected_df["model_type"] = model_type_encoder.fit_transform(selected_df["model_type"])

    selected_df["model_type"] = selected_df["model_type"].astype(int)
    assert selected_df["model_type"].nunique() == 2, "There should be two classes"

    selected_df["cancer_type"] = cancer_type_encoder.fit_transform(selected_df["cancer_type"])
    
    selected_df["cancer_type"] = selected_df["cancer_type"].astype(int)

    print(selected_df.shape)

    # Build VAE
    encoder = build_encoder(feature_count, latent_dim)  # feat count set above, lat dim is a var
    decoder = build_decoder(feature_count, latent_dim)
    vae = VAE(encoder, decoder, columns=selected_df.columns)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), run_eagerly=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = vae.fit(selected_df, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard_callback])

    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(log_dir, "history.tsv", sep = '\t'), index=False)

    # save model
    vae.save(Path(log_dir, "model"))
    # Read-in trained model, to evaluation framework