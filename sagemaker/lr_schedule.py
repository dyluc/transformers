import tensorflow as tf

class WarmupLinearDecay10(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, base_lr):
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.steps_for_warmup = int(0.1 * total_steps) # let's try first 10% of total steps for warmup
        self.steps_for_decay = total_steps - self.steps_for_warmup

    def __call__(self, step):
        # warmup
        if step < self.steps_for_warmup:
            return self.base_lr * (step / self.steps_for_warmup)
        # then linear decay - base LR * (1 - t / T)
        return self.base_lr * (1 - (step - self.steps_for_warmup) / self.steps_for_decay)