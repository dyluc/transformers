import tensorflow as tf

class WarmupLinearDecay10(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, base_lr):
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.steps_for_warmup = int(0.1 * total_steps) # let's try first 10% of total steps for warmup
        self.steps_for_decay = total_steps - self.steps_for_warmup
        # steps for warmup: 14684, steps_for_decay: 132156

    def __call__(self, step):
        # base LR * (1 - t / T)
        step = tf.cast(step, tf.float32)
        return tf.cond(
            step < self.steps_for_warmup,
            lambda: self.base_lr * (step / self.steps_for_warmup), # warmup
            lambda: self.base_lr * (1 - (step - self.steps_for_warmup) / self.steps_for_decay) # then linear decay
        )