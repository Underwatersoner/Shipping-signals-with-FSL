import gc, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, backend as K

from fewshot_utils import (
    CFG, build_vgg16, get_siamese_model,
    load_npy_files_from_folder, get_samples,
    calculate_dist, calculate_dist_val,
    calculate_loss, calculate_acc
)


cfg = CFG()
val_mean_acc = []

for fold in range(1, 51):
    print(f"\n=== Fold {fold} ===")
    random.seed(fold); np.random.seed(fold); tf.random.set_seed(fold)

    # Clear GPU
    K.clear_session(); gc.collect(); time.sleep(1); tf.keras.backend.clear_session()

    # Build model
    base_model = build_vgg16(input_shape=cfg.input_shape)
    model = get_siamese_model(base_model, cfg.input_shape)

    # Data
    train, val = load_npy_files_from_folder(cfg.data_root, fold, cfg)
    print(f"Loaded train/val files: {len(train)}, {len(val)}")

    opt = optimizers.Adam(learning_rate=cfg.lr)

    @tf.function
    def train_step(x1, x2, y_s, y_q):
        with tf.GradientTape() as tape:
            x_out = model([x1, x2])
            dists = calculate_dist(x_out, cfg)
            loss  = calculate_loss(y_s, y_q, *dists)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        acc = calculate_acc(y_q, dists[1])
        return loss, acc

    @tf.function
    def val_step(x1, x2, y_s, y_q):
        x_out = model([x1, x2])
        dists = calculate_dist_val(x_out, cfg)
        loss  = calculate_loss(y_s, y_q, *dists)
        acc   = calculate_acc(y_q, dists[1])
        return loss, acc

    # Training
    for epoch in range(cfg.train_epochs):
        Xl, Xr, Ys, Yq = get_samples(train, cfg)
        loss, acc = train_step(Xl, Xr, Ys, Yq)
        if (epoch + 1) % cfg.train_epochs == 0:
            Vl, Vr, Vs, Vq = get_samples(val, cfg)
            vloss, vacc = val_step(Vl, Vr, Vs, Vq)
            print(f"[{epoch+1:04d}] train loss={loss:.4f} acc={acc:.4f} | val loss={vloss:.4f} acc={vacc:.4f}")

    # Final evaluation
    val_acc_list = []
    for _ in range(cfg.test_epochs):
        Vl, Vr, Vs, Vq = get_samples(val, cfg)
        vloss, vacc = val_step(Vl, Vr, Vs, Vq)
        val_acc_list.append(float(vacc))

    val_mean = float(np.mean(val_acc_list))
    val_mean_acc.append(val_mean)
    print(f"Fold {fold} mean val acc: {val_mean:.4f}")

    pd.DataFrame({"val_mean_acc": val_mean_acc}).to_csv(cfg.csv_name, index=False)
