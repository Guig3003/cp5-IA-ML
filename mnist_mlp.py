"""Experimentos reproduziveis com MLP no MNIST.

Uso:
    python mnist_mlp.py --mode full
    python mnist_mlp.py --mode quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix


SEED = 42
OUT = ROOT / "resultados"
FIG = OUT / "figuras"


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    hidden: tuple[int, ...] = (128,)
    learning_rate: float = 0.001
    batch_size: int = 128
    dropout: float = 0.0
    l2: float = 0.0


def configure_reproducibility() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def load_data(mode: str):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    # Recorte estratificado e deterministico apenas para validar rapidamente o projeto.
    if mode == "quick":
        train_idx = np.concatenate([np.flatnonzero(y_train == d)[:1000] for d in range(10)])
        test_idx = np.concatenate([np.flatnonzero(y_test == d)[:200] for d in range(10)])
        # validation_split separa os ultimos 10%; embaralhar evita que a validacao
        # contenha apenas a ultima classe depois da amostragem estratificada.
        train_idx = np.random.default_rng(SEED).permutation(train_idx)
        x_train, y_train = x_train[train_idx], y_train[train_idx]
        x_test, y_test = x_test[test_idx], y_test[test_idx]
    return x_train, y_train, x_test, y_test


def make_experiments(mode: str) -> list[Experiment]:
    baseline = Experiment("baseline", "baseline")
    widths = (64, 128, 256, 512) if mode == "full" else (64, 128)
    depths = (1, 2, 3) if mode == "full" else (1, 2)
    experiments = [baseline]
    experiments += [
        Experiment(f"arquitetura_d{depth}_w{width}", "arquitetura", (width,) * depth)
        for depth in depths
        for width in widths
    ]
    experiments += [
        Experiment(f"lr_{lr:g}", "learning_rate", learning_rate=lr)
        for lr in ((0.1, 0.01, 0.001) if mode == "full" else (0.01, 0.001))
    ]
    experiments += [
        Experiment(f"batch_{batch}", "batch_size", batch_size=batch)
        for batch in ((32, 64, 128, 256) if mode == "full" else (64, 128))
    ]
    experiments += [
        Experiment("regularizacao_sem", "regularizacao"),
        Experiment("regularizacao_dropout", "regularizacao", dropout=0.3),
        Experiment("regularizacao_l2", "regularizacao", l2=1e-4),
        Experiment("regularizacao_dropout_l2", "regularizacao", dropout=0.3, l2=1e-4),
    ]
    return experiments


def build_model(exp: Experiment) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(exp.l2) if exp.l2 else None
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(784,))]
    for units in exp.hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer))
        if exp.dropout:
            layers.append(tf.keras.layers.Dropout(exp.dropout))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    model = tf.keras.Sequential(layers, name=exp.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=exp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def save_confusion(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.title(f"Matriz de confusao — {name}")
    plt.tight_layout()
    plt.savefig(FIG / f"confusao_{safe_name(name)}.png", dpi=150)
    plt.close()


def save_curves(history: dict[str, list[float]], name: str) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["accuracy"], label="treino")
    axes[0].plot(epochs, history["val_accuracy"], label="validacao")
    axes[0].set(title="Acuracia", xlabel="Epoca", ylabel="Acuracia")
    axes[1].plot(epochs, history["loss"], label="treino")
    axes[1].plot(epochs, history["val_loss"], label="validacao")
    axes[1].set(title="Perda", xlabel="Epoca", ylabel="Loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(FIG / f"curvas_{safe_name(name)}.png", dpi=150)
    plt.close(fig)


def top_confusions(y_true, y_pred, limit: int = 3) -> list[dict[str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm.ravel())[::-1][:limit]
    return [
        {"real": int(i // 10), "predito": int(i % 10), "quantidade": int(cm.ravel()[i])}
        for i in flat
    ]


def run_experiment(exp, x_train, y_train, x_test, y_test, max_epochs: int):
    tf.keras.backend.clear_session()
    configure_reproducibility()
    model = build_model(exp)
    start = time.perf_counter()
    history_obj = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=max_epochs,
        batch_size=exp.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=2,
        shuffle=True,
    )
    elapsed = time.perf_counter() - start
    history = history_obj.history
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=512, verbose=0).argmax(axis=1)
    save_confusion(y_test, y_pred, exp.name)
    save_curves(history, exp.name)
    model.save(OUT / f"modelo_{safe_name(exp.name)}.keras") if exp.name == "baseline" else None

    row = asdict(exp)
    row["hidden"] = "-".join(map(str, exp.hidden))
    row.update(
        epochs_run=len(history["loss"]),
        parameters=model.count_params(),
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        generalization_gap=float(train_acc - test_acc),
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        best_val_accuracy=float(max(history["val_accuracy"])),
        final_val_loss=float(history["val_loss"][-1]),
        time_seconds=elapsed,
        top_confusions=json.dumps(top_confusions(y_test, y_pred), ensure_ascii=False),
    )
    return row, model, y_pred


def save_error_examples(model, x_test, y_test) -> None:
    probs = model.predict(x_test, batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(pred != y_test)
    wrong = wrong[np.argsort(probs[wrong, pred[wrong]])[::-1]][:20]
    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    for ax, idx in zip(axes.flat, wrong):
        ax.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"real {y_test[idx]} / pred {pred[idx]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Erros mais confiantes do melhor modelo")
    fig.tight_layout()
    fig.savefig(FIG / "erros_melhor_modelo.png", dpi=160)
    plt.close(fig)


def save_summary_plot(df: pd.DataFrame) -> None:
    ordered = df.sort_values("test_accuracy", ascending=True)
    plt.figure(figsize=(10, max(6, len(df) * 0.3)))
    sns.barplot(data=ordered, x="test_accuracy", y="name", hue="group", dodge=False)
    plt.xlim(max(0.7, ordered.test_accuracy.min() - 0.02), 1.0)
    plt.xlabel("Acuracia no teste")
    plt.ylabel("Experimento")
    plt.title("Comparacao de todos os experimentos")
    plt.tight_layout()
    plt.savefig(FIG / "comparacao_geral.png", dpi=160)
    plt.close()


def build_report(df: pd.DataFrame, mode: str) -> str:
    best = df.loc[df.test_accuracy.idxmax()]
    baseline = df[df.name == "baseline"].iloc[0]
    most_overfit = df.loc[df.generalization_gap.idxmax()]
    lr_rows = df[df.group == "learning_rate"].sort_values("learning_rate")
    batch_rows = df[df.group == "batch_size"].sort_values("batch_size")
    reg_rows = df[df.group == "regularizacao"].sort_values("test_accuracy", ascending=False)
    confusions = json.loads(best.top_confusions)
    confusion_text = ", ".join(
        f"{c['real']}→{c['predito']} ({c['quantidade']} casos)" for c in confusions
    )
    overfit = df[df.generalization_gap > 0.03]

    def table(rows, cols):
        return rows[cols].to_markdown(index=False, floatfmt=".4f")

    architecture = df[df.group == "arquitetura"].copy()
    architecture["profundidade"] = architecture.hidden.str.count("-") + 1
    architecture["largura"] = architecture.hidden.str.split("-").str[0].astype(int)

    return f"""# Relatório — MNIST com Redes Neurais MLP

## Metodologia

Os pixels foram normalizados de `[0, 255]` para `[0, 1]` e cada imagem 28×28 foi transformada em um vetor de 784 atributos. Todas as camadas ocultas usam ReLU, a saída usa softmax e a função de perda é entropia cruzada categórica esparsa. Foi usado Adam, semente {SEED}, 10% do treino para validação e early stopping (paciência 3, restaurando os melhores pesos). O conjunto de teste não participa do ajuste.

Esta execução usou o modo **{mode}**. {"O modo quick usa uma amostra estratificada e serve para validar o pipeline; as conclusões finais devem ser produzidas com `--mode full`." if mode == "quick" else "O modo full usa as 60.000 imagens de treino e 10.000 de teste."}

## Resultado geral

A baseline obteve **{baseline.test_accuracy:.4%}** no teste. A melhor configuração foi **{best['name']}**, com **{best.test_accuracy:.4%}** no teste, **{best.train_accuracy:.4%}** no treino e gap de **{best.generalization_gap:.4%}**.

![Comparação geral](figuras/comparacao_geral.png)

## Arquitetura: profundidade e largura

{table(architecture.sort_values(['profundidade', 'largura']), ['name', 'hidden', 'parameters', 'train_accuracy', 'test_accuracy', 'generalization_gap'])}

Mais camadas e neurônios elevam a capacidade e o custo computacional, mas não garantem melhoria no teste. O ponto de saturação pode ser observado quando o número de parâmetros continua crescendo sem ganho proporcional de acurácia; a tabela permite identificá-lo diretamente nesta execução.

## Learning rate

{table(lr_rows, ['learning_rate', 'epochs_run', 'best_val_accuracy', 'test_accuracy', 'final_val_loss'])}

Learning rates altos podem oscilar ou convergir para uma solução pior; valores muito baixos tendem a exigir mais épocas. A combinação mais estável nesta varredura foi a que conciliou maior acurácia de validação com perda final baixa: **{lr_rows.loc[lr_rows.best_val_accuracy.idxmax(), 'name'] if len(lr_rows) else 'não avaliado'}**.

## Batch size

{table(batch_rows, ['batch_size', 'epochs_run', 'best_val_accuracy', 'test_accuracy', 'time_seconds'])}

Batches menores fazem mais atualizações e introduzem mais ruído; batches maiores reduzem o número de atualizações por época e podem ser mais rápidos, mas nem sempre generalizam melhor.

## Regularização e overfitting

{table(reg_rows, ['name', 'dropout', 'l2', 'train_accuracy', 'test_accuracy', 'generalization_gap'])}

Foi adotado como sinal operacional de overfitting um gap treino–teste acima de 3 pontos percentuais. **{len(overfit)} de {len(df)}** configurações ultrapassaram esse limiar. O maior gap foi o de **{most_overfit['name']}** ({most_overfit.generalization_gap:.4%}). Dropout desativa aleatoriamente 30% das ativações durante o treino; L2 penaliza pesos altos com fator `1e-4`. Se a regularização reduz o gap sem perda relevante no teste, ela melhorou a generalização; regularização excessiva aparece como baixa acurácia também no treino (underfitting).

## Matriz de confusão e análise dos erros

No melhor modelo, as confusões direcionais mais frequentes foram: **{confusion_text}**. Isso indica quais formas manuscritas ficaram visualmente mais ambíguas. Há uma matriz e curvas individuais para **cada** experimento em `resultados/figuras/`.

![Matriz do melhor modelo](figuras/confusao_{safe_name(str(best['name']))}.png)

![Exemplos de erros](figuras/erros_melhor_modelo.png)

## Conclusão crítica

- Melhor desempenho: **{best['name']}**, acurácia de teste **{best.test_accuracy:.4%}**.
- Overfitting: avaliado pelo gap treino–teste e pela divergência entre as curvas de treino e validação; o caso mais forte foi **{most_overfit['name']}**.
- Estabilidade: **{lr_rows.loc[lr_rows.best_val_accuracy.idxmax(), 'name'] if len(lr_rows) else 'não avaliado'}** apresentou o melhor pico de validação entre os learning rates comparados.
- Saturação: deve ser considerada a menor arquitetura cuja acurácia fica próxima da melhor (por exemplo, diferença menor que 0,2 p.p.), pois redes maiores aumentam parâmetros e tempo sem benefício prático garantido.

Os valores completos e sem arredondamento estão em `metricas.csv`; as configurações estão em `configuracoes.json`.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Experimentos MLP no MNIST")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=None, help="Máximo de épocas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_epochs = args.epochs or (8 if args.mode == "quick" else 20)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    configure_reproducibility()
    x_train, y_train, x_test, y_test = load_data(args.mode)
    experiments = make_experiments(args.mode)
    (OUT / "configuracoes.json").write_text(
        json.dumps([asdict(e) for e in experiments], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    best_acc = -1.0
    best_model = None
    for index, exp in enumerate(experiments, 1):
        print(f"\n[{index}/{len(experiments)}] {exp.name}")
        row, model, _ = run_experiment(exp, x_train, y_train, x_test, y_test, max_epochs)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "metricas_parciais.csv", index=False)
        if row["test_accuracy"] > best_acc:
            best_acc = row["test_accuracy"]
            best_model = model

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metricas.csv", index=False)
    save_summary_plot(df)
    assert best_model is not None
    save_error_examples(best_model, x_test, y_test)
    (OUT / "relatorio.md").write_text(build_report(df, args.mode), encoding="utf-8")
    print(f"\nConcluído. Relatório: {OUT / 'relatorio.md'}")


if __name__ == "__main__":
    main()

