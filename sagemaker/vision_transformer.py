import tensorflow as tf

class PatchConverter(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dim = patches.shape[-1] # last dimension of patches is the flattened patch size (e.g. 16x16 patches is 768 - 16x16x3)
        patches = tf.reshape(patches, [batch_size, -1, patch_dim]) # reshape patches into 3D tensor of shape [batch_size, total patches per image, flattened patch size]
        return patches


class PatchEmbeddingWithClassToken(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Dense(embedding_dim) # a simple dense layer for projection
        self.class_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="class_token"
        )

    def call(self, patches):
        patch_embeddings = self.embedding_layer(patches)
        batch_size = tf.shape(patch_embeddings)[0]
        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, self.class_token.shape[-1]]) # duplicate class_token to match batch size
        return tf.concat([class_token, patch_embeddings], axis=1)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="positional_embeddings"
        )

    def call(self, patch_embeddings):
        embeddings = self.position_embeddings + patch_embeddings
        return embeddings


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn_layer = tf.keras.layers.MultiHeadAttention( # key_dim = 768 / 12 = 64
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation="gelu"), # paper user GELU activation
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # mutli-head attention sublayer
        x = self.norm1(inputs)
        x = self.attn_layer(x, value=x, training=training)
        x = self.dropout(x, training=training)
        x = tf.keras.layers.Add()([x, inputs])
        
        # mlp sublayer
        x1 = self.norm2(x)
        x1 = self.mlp(x1, training=training)
        x1 = self.dropout(x1, training=training)
        x1 = tf.keras.layers.Add()([x1, x])

        return x1


class ClassificationHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, mlp_dim=None, **kwargs):
        super().__init__(**kwargs)
        output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
        if mlp_dim:
            self.mlp = tf.keras.Sequential([ # can add dropout here if overfitting during training
                tf.keras.layers.Dense(mlp_dim, activation="gelu"),
                output_layer
            ])
        else:
            self.mlp = output_layer

    def call(self, class_token):
        return self.mlp(class_token)


class ViT(tf.keras.Model):
    def __init__(
        self, 
        input_shape, 
        patch_size, 
        num_classes, 
        embedding_dim, 
        num_heads, 
        num_layers, 
        mlp_dim, 
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_converter = PatchConverter(patch_size)
        self.patch_embedding_class_token = PatchEmbeddingWithClassToken(embedding_dim)
        self.positional_embedding = PositionalEmbedding(
            num_patches=(input_shape[0] // patch_size) ** 2 + 1, # +1 includes [class] embedding
            embedding_dim=embedding_dim
        )
        self.encoder_stack = [
            EncoderLayer(num_heads, embedding_dim, mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.class_token_norm = tf.keras.layers.LayerNormalization()
        self.classification_head = ClassificationHead(num_classes, 2048)


    def call(self, inputs):
        # Extract the patches and create the patch embeddings w/ class token
        patches = self.patch_converter(inputs)
        embeddings = self.patch_embedding_class_token(patches)

        # Add positional embeddings
        embeddings = self.positional_embedding(embeddings)

        # Encoder stack
        x = embeddings
        for layer in self.encoder_stack:
            x = layer(x)

        # Extract and normalize the [class] token
        class_token = x[:, 0]
        norm_class_token = self.class_token_norm(class_token)

        # Classification head
        class_probas = self.classification_head(norm_class_token)
        
        return class_probas
