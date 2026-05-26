import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Concatenate, \
    Dropout


def generator_model(latent_dim, n_channels, patch_size, condition_manager):
    """
    Generator model for 3D meniscus tissue generation (multi-channel output).

    Generates 10 depth slices simultaneously as independent channels.
    Each channel represents one slice through the tissue volume.

    Args:
        latent_dim (int): Noise vector dimension (e.g., 100)
        n_channels (int): Number of output channels (should be 10 for 3D)
        patch_size (int): Output spatial size (256 or 512)
        condition_manager: Manager with active conditions (FVF, alignment, etc.)

    Returns:
        Model: Generator with output shape (batch, patch_size, patch_size, n_channels)
               Values in [-1, 1] range (tanh activation)
    """

    # Validation
    assert n_channels >= 1, f"n_channels must be >= 1, got {n_channels}"
    assert patch_size % 16 == 0, f"patch_size must be divisible by 16, got {patch_size}"

    # Dynamic filter scaling: keeps current values for n_channels<=32,
    # widens automatically for larger n_channels (e.g. 64)
    g_filters = max(64, n_channels * 2)  # 32ch?64, 64ch?128

    # Noise input
    noise_input = Input(shape=(latent_dim,), name='noise_input')

    # Condition inputs and embeddings
    condition_inputs = []
    condition_embeddings = []
    initial_size = patch_size // 16

    for condition in condition_manager.active_conditions:
        cond_input = Input(shape=(1,), dtype='float32', name=f'{condition.name}_input')
        condition_inputs.append(cond_input)

        # Spatially tile condition
        li = Reshape((1, 1, 1))(cond_input)
        li = tf.tile(li, [1, initial_size, initial_size, 1])
        condition_embeddings.append(li)

    # Generate from noise
    gen = Dense(initial_size * initial_size * 512)(noise_input)
    gen = LeakyReLU(0.2)(gen)
    gen = Reshape((initial_size, initial_size, 512))(gen)

    # Combine with conditions
    if condition_embeddings:
        gen = Concatenate()([gen] + condition_embeddings)

    # Transposed convolution blocks (upsample 2x each)
    x = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same')(gen)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(g_filters, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(g_filters * 2, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(g_filters * 2, (3, 3), padding='same')(x)
    x = LeakyReLU(0.2)(x)

    # Final output: 10 channels for 10 depth slices
    generated_output = Conv2DTranspose(n_channels, (5, 5), strides=(2, 2),
                                       padding='same', activation='tanh')(x)

    # Create model
    model_inputs = [noise_input] + condition_inputs
    model = Model(inputs=model_inputs, outputs=generated_output, name='Generator_3D')

    # Print model info
    print(f"\n{'=' * 70}")
    print(f"3D GENERATOR CREATED")
    print(f"{'=' * 70}")
    print(f"  Output shape: (batch, {patch_size}, {patch_size}, {n_channels})")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Conditions: {len(condition_inputs)} active")
    print(f"  Parameters: {model.count_params():,}")
    print(f"{'=' * 70}\n")

    return model


def discriminator_model(in_shape, condition_manager):
    """
    Discriminator model for 3D meniscus tissue GAN (multi-channel input).

    Processes 10 depth slices simultaneously through the channel dimension.
    First Conv2D layer sees all slices at each spatial position.

    Args:
        in_shape (tuple): Input shape (H, W, C) - should be (256, 256, 10)
        condition_manager: Manager with active conditions

    Returns:
        Model: Discriminator with output shape (batch, 1) - real/fake probability
               Values in [0, 1] range (sigmoid activation)
    """

    # Validation
    assert len(in_shape) == 3, f"in_shape must be (H, W, C), got {in_shape}"
    height, width, channels = in_shape
    assert channels >= 1, f"channels must be >= 1, got {channels}"
    assert height == width, f"Height and width must match, got {height}x{width}"
    assert height % 32 == 0 and height >= 128, f"Height must be divisible by 32 and >= 128, got {height}"

    # Dynamic filter scaling: ensures first layer always expands the input
    # 32ch?64 (same as before), 64ch?128 (avoids compression bottleneck)
    d_filters = max(64, channels * 2)

    # Image input
    image_input = Input(shape=in_shape, name='image_input')

    # Condition inputs and embeddings
    condition_inputs = []
    condition_embeddings = []

    for condition in condition_manager.active_conditions:
        cond_input = Input(shape=(1,), dtype='float32', name=f'{condition.name}_input')
        condition_inputs.append(cond_input)

        # Spatially tile condition
        li = Reshape((1, 1, 1))(cond_input)
        li = tf.tile(li, [1, in_shape[0], in_shape[1], 1])
        condition_embeddings.append(li)

    # Combine image with conditions
    if condition_embeddings:
        x = Concatenate()([image_input] + condition_embeddings)
    else:
        x = image_input

    # Depth-aware preprocessing for 3D (filter counts scale with n_channels)
    x = Conv2D(d_filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(d_filters * 2, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    # Classification
    x = Flatten()(x)
    validity_output = Dense(1, activation='sigmoid', name='validity_output')(x)

    # Create model
    model_inputs = [image_input] + condition_inputs
    model = Model(inputs=model_inputs, outputs=validity_output, name='Discriminator_3D')

    # Print model info
    print(f"\n{'=' * 70}")
    print(f"3D DISCRIMINATOR CREATED")
    print(f"{'=' * 70}")
    print(f"  Input shape: (batch, {in_shape[0]}, {in_shape[1]}, {in_shape[2]})")
    print(f"  Conditions: {len(condition_inputs)} active")
    print(f"  First layer: (3, 3) stride 2 -> {d_filters} filters (dynamic scaling)")
    print(f"  Parameters: {model.count_params():,}")
    print(f"{'=' * 70}\n")

    return model