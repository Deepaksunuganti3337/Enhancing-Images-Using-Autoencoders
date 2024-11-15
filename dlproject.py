# -*- coding: utf-8 -*-
"""DLProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pAwZafTBAeLHKodbU5Y34S6CqU_ZP_SA

**Importing Libraries**
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

"""**Loading the Dataset**"""

dataset_dir = '/content/drive/MyDrive/low_res'

"""**Preprocessing the images**"""

def load_and_preprocess_images(directory, target_size=(64, 64)):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=target_size)  # Resize image
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
            filenames.append(filename)
    return np.array(images), filenames

"""**Building The AutoEncoder Model**"""

def build_autoencoder(input_shape=(64, 64, 3)):
    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)

    # Compile model
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder

"""**Function to calculate accuracy metrics (MSE, SSIM, PSNR)**

"""

def calculate_metrics(original_images, enhanced_images):
    mse_list = []
    ssim_list = []
    psnr_list = []

    for orig, enhanced in zip(original_images, enhanced_images):
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(orig.flatten(), enhanced.flatten())
        mse_list.append(mse)

        # Calculate Structural Similarity Index (SSIM)
        ssim_value = ssim(orig, enhanced, multichannel=True, win_size=3, data_range=1.0, channel_axis=2)
        ssim_list.append(ssim_value)

        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        psnr_value = psnr(orig, enhanced, data_range=1.0)
        psnr_list.append(psnr_value)

    avg_mse = np.mean(mse_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean(psnr_list)

    return avg_mse, avg_ssim, avg_psnr, mse_list, ssim_list, psnr_list

"""**Function to display original and enhanced images**"""

def display_images(original_images, enhanced_images, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Display original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i])
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display enhanced images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(enhanced_images[i])
        plt.title("Enhanced")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

"""**Function to train the autoencoder**"""

def train_autoencoder(autoencoder, x_train, epochs=200, batch_size=4):
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)
    return history

"""**filter overfitting images based on MSE and SSIM thresholds**"""

def filter_overfitting_images(mse_list, ssim_list, mse_threshold=0.05, ssim_threshold=0.8):
    valid_indices = [i for i, (mse, ssim) in enumerate(zip(mse_list, ssim_list)) if mse < mse_threshold and ssim > ssim_threshold]
    return valid_indices

"""**Main Function**"""

# Main function
def main():
    # Load the dataset (all images from the directory)
    x_train, filenames_train = load_and_preprocess_images(dataset_dir, target_size=(64, 64))

    print(f"Dataset loaded with {x_train.shape[0]} images, each of size {x_train.shape[1:]}")

    # Build the autoencoder model
    autoencoder = build_autoencoder(input_shape=x_train.shape[1:])

    # Train the autoencoder model
    history = train_autoencoder(autoencoder, x_train, epochs=200, batch_size=4)

    # Enhance the images using the trained autoencoder
    enhanced_images = autoencoder.predict(x_train)

    # Calculate MSE, SSIM, and PSNR for all images
    avg_mse, avg_ssim, avg_psnr, mse_list, ssim_list, psnr_list = calculate_metrics(x_train, enhanced_images)

    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

    # Filter out overfitting images
    valid_indices = filter_overfitting_images(mse_list, ssim_list)
    print(f"Valid Images after filtering: {len(valid_indices)} images")

    # Create a table with evaluation metrics for the first 10 images (or valid ones)
    table_data = []
    for i in range(min(10, len(valid_indices))):
        index = valid_indices[i]
        table_data.append({
            "Image": filenames_train[index],
            "MSE": mse_list[index],
            "SSIM": ssim_list[index],
            "PSNR": psnr_list[index]
        })

    # Convert the table to a DataFrame
    df = pd.DataFrame(table_data)
    print(df)

    # Display original and enhanced images
    display_images(x_train[valid_indices], enhanced_images[valid_indices], num_images=min(10, len(valid_indices)))

    # Plotting training history
    if 'loss' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.title("Autoencoder Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.show()

    # Plot MSE, SSIM, and PSNR comparison for each image
    plt.figure(figsize=(12, 4))

    # MSE Plot
    plt.subplot(1, 3, 1)
    plt.plot(mse_list, label="MSE", color='red')
    plt.title("Mean Squared Error (MSE)")
    plt.xlabel("Image Index")
    plt.ylabel("MSE")
    plt.grid(True)

    # SSIM Plot
    plt.subplot(1, 3, 2)
    plt.plot(ssim_list, label="SSIM", color='green')
    plt.title("Structural Similarity Index (SSIM)")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.grid(True)

    # PSNR Plot
    plt.subplot(1, 3, 3)
    plt.plot(psnr_list, label="PSNR", color='blue')
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot histograms for MSE, SSIM, and PSNR
    plt.figure(figsize=(18, 6))

    # MSE Histogram
    plt.subplot(1, 3, 1)
    plt.hist(mse_list, bins=20, color='red', alpha=0.7)
    plt.title("Histogram of MSE")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")

    # SSIM Histogram
    plt.subplot(1, 3, 2)
    plt.hist(ssim_list, bins=20, color='green', alpha=0.7)
    plt.title("Histogram of SSIM")
    plt.xlabel("SSIM")
    plt.ylabel("Frequency")

    # PSNR Histogram
    plt.subplot(1, 3, 3)
    plt.hist(psnr_list, bins=20, color='blue', alpha=0.7)
    plt.title("Histogram of PSNR")
    plt.xlabel("PSNR")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()