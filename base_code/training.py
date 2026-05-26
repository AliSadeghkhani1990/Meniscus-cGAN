import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from base_code.data_processing import read_patch, read_patch_3D
from base_code.utils import plot_generated_images_enhanced


# ============================================================================
# INTER-SLICE CONTINUITY LOSS (3D)
# ============================================================================

def inter_slice_continuity_loss(generated_volume):
    slices_current = generated_volume[:, :, :, :-1]
    slices_next = generated_volume[:, :, :, 1:]
    differences = tf.abs(slices_current - slices_next)
    continuity_loss = tf.reduce_mean(differences)
    return continuity_loss


def generate_real_samples(original_volumes, balanced_patch_info, n_samples, patch_size, condition_manager, n_slices):
    """
    Generate real samples with their condition values (3D version).

    Args:
        original_volumes: List of 3D volumes (H, W, D)
        balanced_patch_info: List of patch metadata
        n_samples: Number of samples to generate
        patch_size: Spatial size (256)
        condition_manager: Condition manager
        n_slices: Number of depth slices (10)

    Returns:
        Tuple of ([images, cond1, cond2, ...], labels)
        where images shape is (n_samples, patch_size, patch_size, n_slices)
    """
    indices = np.random.randint(0, len(balanced_patch_info), n_samples)

    # Use read_patch_3D for 3D volumes
    selected_patches = [
        read_patch_3D(original_volumes, balanced_patch_info[i], patch_size, n_slices)
        for i in indices
    ]

    selected_X = np.array(selected_patches, dtype=np.float32)
    selected_X = (selected_X / 255.0 - 0.5) * 2  # Scale to [-1, 1]

    # Get condition values for each active condition
    condition_inputs = []
    for condition in condition_manager.active_conditions:
        condition_values = [balanced_patch_info[i][f'{condition.name}_value']
                            for i in indices]
        condition_inputs.append(np.array(condition_values, dtype=np.float32))

    return [selected_X] + condition_inputs, np.ones((n_samples, 1))


def generate_latent_points(latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes):
    """Generate latent points and condition inputs for GAN."""
    z_input = np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)

    if not condition_manager.active_conditions:
        return [z_input]

    # Generate random condition values based on the dataset distribution
    condition_values = []

    for condition in condition_manager.active_conditions:
        # Get all unique values for this condition from the dataset
        values = [patch[f'{condition.name}_value'] for patch in balanced_patch_info]
        unique_values = sorted(set(values))

        # Randomly sample from the available values
        selected_values = np.random.choice(unique_values, size=n_samples)
        condition_values.append(selected_values.astype(np.float32))

    return [z_input] + condition_values


def generate_fake_samples(generator, latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes):
    """Generate fake samples using the generator."""
    # Get latent points and condition inputs
    inputs = generate_latent_points(latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes)

    # Generate images
    images = generator.predict(inputs)

    # Create labels for discriminator (all fake = 0)
    y = np.zeros((n_samples, 1))

    # Return images with condition inputs
    return [images] + inputs[1:] if len(inputs) > 1 else [images], y


def train(g_model, d_model, dataset, balanced_patch_info, latent_dim, epochs, n_batch,
          saving_step, non_empty_classes, patch_size, model_path_saving, n_classes_category, n_classes_condition,
          save_interval, last_epochs_to_save, bat_per_epo, n_channels, condition_manager, n_slices, d_slices=16, start_epoch=0):
    """
    Train the conditional GAN for meniscus tissue generation.
    """
    original_volumes = dataset[0]

    gen_losses = []
    disc_losses = []
    real_scores = []
    fake_scores = []
    continuity_losses = []
    adversarial_losses = []

    # Calculate total steps for the learning rate schedule
    total_steps = epochs * bat_per_epo

    # Initialize learning rate schedules
    lr_scheduleG = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=2e-4,
        decay_steps=total_steps,
        end_learning_rate=2e-6,
        power=1.0
    )

    lr_scheduleD = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=2e-4,
        decay_steps=total_steps,
        end_learning_rate=2e-6,
        power=1.0
    )

    # Initialize optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduleG, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduleD, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

    # Define loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    if start_epoch > 0:
        print(f"\n?? CONTINUING TRAINING")
        print(f"   Starting from epoch: {start_epoch + 1}")
        print(f"   Target epochs: {epochs}")
        print(f"   Remaining: {epochs - start_epoch} epochs\n")

    @tf.function
    def train_step(real_samples_and_conditions):
        """Single training step for both generator and discriminator."""
        real_images = real_samples_and_conditions[0]
        condition_inputs = real_samples_and_conditions[1:]
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, latent_dim])

        # Random consecutive block of 16 slices for discriminator
        max_start = tf.shape(real_images)[3] - d_slices
        r1 = tf.random.uniform([], 0.0, 1.0)
        r2 = tf.random.uniform([], 0.0, 1.0)
        start_idx = tf.cast(tf.math.minimum(r1, r2) * tf.cast(max_start + 1, tf.float32), tf.int32)
        start_idx = tf.minimum(start_idx, max_start)
        if_flip = tf.random.uniform([], 0.0, 1.0) > 0.5
        start_idx = tf.cond(if_flip, lambda: start_idx, lambda: max_start - start_idx)

        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            generator_inputs = [noise] + list(condition_inputs)
            generated_images = g_model(generator_inputs, training=True)

            # Slice both real and fake to 16 channels
            real_sliced = real_images[:, :, :, start_idx:start_idx + d_slices]
            fake_sliced = generated_images[:, :, :, start_idx:start_idx + d_slices]

            discriminator_real_inputs = [real_sliced] + list(condition_inputs)
            discriminator_fake_inputs = [fake_sliced] + list(condition_inputs)

            real_output = d_model(discriminator_real_inputs, training=True)
            fake_output = d_model(discriminator_fake_inputs, training=True)

            d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = disc_tape.gradient(d_loss, d_model.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))

        # Train Generator (new random slice)
        noise2 = tf.random.normal([batch_size, latent_dim])
        r3 = tf.random.uniform([], 0.0, 1.0)
        r4 = tf.random.uniform([], 0.0, 1.0)
        start_idx2 = tf.cast(tf.math.minimum(r3, r4) * tf.cast(max_start + 1, tf.float32), tf.int32)
        start_idx2 = tf.minimum(start_idx2, max_start)
        if_flip2 = tf.random.uniform([], 0.0, 1.0) > 0.5
        start_idx2 = tf.cond(if_flip2, lambda: start_idx2, lambda: max_start - start_idx2)

        with tf.GradientTape() as gen_tape:
            generator_inputs = [noise2] + list(condition_inputs)
            generated_images = g_model(generator_inputs, training=True)

            fake_sliced2 = generated_images[:, :, :, start_idx2:start_idx2 + d_slices]

            discriminator_inputs = [fake_sliced2] + list(condition_inputs)
            fake_output = d_model(discriminator_inputs, training=False)
            g_loss_adversarial = loss_fn(tf.ones_like(fake_output), fake_output)
            continuity_loss = inter_slice_continuity_loss(generated_images)
            lambda_continuity = 0.1
            g_loss = g_loss_adversarial + lambda_continuity * continuity_loss

        g_gradients = gen_tape.gradient(g_loss, g_model.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))

        return d_loss, g_loss, g_loss_adversarial, continuity_loss

    # Training loop
    fixed_noise = None
    fixed_conditions = None

    progress_bar = tqdm(range(start_epoch, epochs), desc='Epochs', initial=start_epoch, total=epochs)
    for i in progress_bar:
        total_gen_loss = 0
        total_disc_loss = 0
        total_continuity_loss = 0  # NEW
        total_adversarial_loss = 0  # NEW

        for j in range(bat_per_epo):
            # Get real samples with conditions
            real_samples_and_labels = generate_real_samples(
                original_volumes, balanced_patch_info, n_batch, patch_size, condition_manager, n_slices
            )
            real_samples, y_real = real_samples_and_labels

            # Convert all inputs to tensors
            real_samples = [tf.convert_to_tensor(x, dtype=tf.float32) for x in real_samples]

            # Perform training step (now returns 4 values)
            d_loss, g_loss, g_loss_adv, cont_loss = train_step(real_samples)

            total_disc_loss += d_loss
            total_gen_loss += g_loss
            total_adversarial_loss += g_loss_adv  # NEW
            total_continuity_loss += cont_loss  # NEW

        # Calculate means
        mean_gen_loss = total_gen_loss / bat_per_epo
        mean_disc_loss = total_disc_loss / bat_per_epo
        mean_continuity_loss = total_continuity_loss / bat_per_epo  # NEW
        mean_adversarial_loss = total_adversarial_loss / bat_per_epo  # NEW

        gen_losses.append(mean_gen_loss)
        disc_losses.append(mean_disc_loss)
        continuity_losses.append(mean_continuity_loss)  # NEW
        adversarial_losses.append(mean_adversarial_loss)  # NEW


        # Update progress bar
        progress_bar.set_postfix({
            'gen_loss': mean_gen_loss.numpy(),
            'disc_loss': mean_disc_loss.numpy(),
            'continuity': mean_continuity_loss.numpy()  # NEW
        })

        if (i + 1) % 10 == 0:  # Print every 10 epochs
            print(f"\nEpoch {i + 1}/{epochs}")
            print(f"Generator loss: {mean_gen_loss.numpy():.4f}")
            print(f"  - Adversarial: {mean_adversarial_loss.numpy():.4f}")  # NEW
            print(f"  - Continuity: {mean_continuity_loss.numpy():.4f}")  # NEW
            print(f"Discriminator loss: {mean_disc_loss.numpy():.4f}")

        # Evaluate discriminator performance
        [X_real, *condition_inputs], y_real = generate_real_samples(
            original_volumes, balanced_patch_info, n_batch, patch_size, condition_manager, n_slices
        )
        [X_fake, *condition_inputs_fake], y_fake = generate_fake_samples(
            g_model, latent_dim, n_batch, condition_manager, balanced_patch_info, non_empty_classes
        )

        # Slice to 16 channels for evaluation
        r1, r2 = np.random.rand(), np.random.rand()
        eval_start = int(min(r1, r2) * (X_real.shape[3] - d_slices + 1))
        if np.random.rand() > 0.5:
            eval_start = (X_real.shape[3] - d_slices) - eval_start
        X_real_sliced = X_real[:, :, :, eval_start:eval_start + d_slices]
        X_fake_sliced = X_fake[:, :, :, eval_start:eval_start + d_slices]

        real_inputs = [X_real_sliced] + condition_inputs
        fake_inputs = [X_fake_sliced] + condition_inputs_fake

        real_score = d_model.evaluate(real_inputs, y_real, verbose=0)[1]
        fake_score = d_model.evaluate(fake_inputs, y_fake, verbose=0)[1]
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Save models logic
        save_current_epoch = (i >= epochs - last_epochs_to_save) or (save_interval and (i + 1) % saving_step == 0)
        if save_current_epoch:
            g_model.save(os.path.join(model_path_saving, f'G_epoch_{i + 1}.h5'), save_format='tf',
                         include_optimizer=True)
            d_model.save(os.path.join(model_path_saving, f'D_epoch_{i + 1}.h5'), save_format='tf',
                         include_optimizer=True)
            print(f"Models saved at epoch {i + 1}")

        # Generate and visualize samples
        fixed_noise, fixed_conditions = plot_generated_images_enhanced(
            g_model, i + 1, latent_dim, n_channels, condition_manager, balanced_patch_info,
            fixed_noise, fixed_conditions
        )

    # Plot training results
    plot_training_results(gen_losses, disc_losses, real_scores, fake_scores, epochs,
                      continuity_losses, adversarial_losses, start_epoch=start_epoch)


def plot_training_results(gen_losses, disc_losses, real_scores, fake_scores, epochs,
                          continuity_losses=None, adversarial_losses=None, start_epoch=0):
    """Plot training curves including continuity loss for 3D mode."""

    # Determine number of subplots
    n_plots = 3 if continuity_losses is not None else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if n_plots == 2:
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    epoch_range = range(start_epoch + 1, start_epoch + 1 + len(gen_losses))

    # Plot 1: Generator and Discriminator Loss
    ax1.set_xlabel('Epoch')
    ax1.set_title('Generator and Discriminator Loss')

    color = 'tab:red'
    ax1.set_ylabel('Generator Loss', color=color)
    ax1.plot(epoch_range, gen_losses, color=color, label='Generator Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1_twin = ax1.twinx()
    color = 'tab:blue'
    ax1_twin.set_ylabel('Discriminator Loss', color=color)
    ax1_twin.plot(epoch_range, disc_losses, color=color, label='Discriminator Loss')
    ax1_twin.tick_params(axis='y', labelcolor=color)

    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Real and Fake Accuracy
    ax2.plot(epoch_range, real_scores, label='Real Score')
    ax2.plot(epoch_range, fake_scores, label='Fake Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Real and Fake Accuracy Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss Components (3D mode only)
    if continuity_losses is not None:
        ax3.plot(epoch_range, adversarial_losses,
                 label='Adversarial Loss', color='orange', linewidth=2)
        ax3.plot(epoch_range, continuity_losses,
                 label='Continuity Loss', color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Generator Loss Components (3D Mode)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    try:
        from __main__ import plot_dirs
        plt.savefig(os.path.join(plot_dirs['evaluation_plots'], 'training_curves.png'),
                    dpi=300, bbox_inches='tight')
    except:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

    plt.close()