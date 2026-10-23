# Análise do MNIST com MLP

Projeto experimental para avaliar como profundidade, largura, learning rate, batch size, dropout e regularização L2 afetam uma rede MLP no MNIST.

## O que o projeto entrega

- baseline MLP;
- 1, 2 e 3 camadas ocultas, com 64, 128, 256 e 512 neurônios;
- learning rates 0.1, 0.01 e 0.001;
- batches 32, 64, 128 e 256;
- comparação sem regularização, com dropout, L2 e ambos;
- acurácia de treino, validação e teste, loss, tempo e gap de generalização;
- matriz de confusão e curvas para cada experimento;
- exemplos dos erros mais confiantes do melhor modelo;
- relatório crítico em Markdown gerado com os resultados reais.

## Como o código funciona

O arquivo `mnist_mlp.py` executa o fluxo completo:

1. baixa o MNIST pelo TensorFlow e normaliza os pixels para o intervalo `[0, 1]`;
2. transforma cada imagem 28×28 em um vetor com 784 atributos;
3. reserva 10% do conjunto de treino para validação;
4. cria uma MLP com camadas densas, ativação ReLU e saída softmax;
5. treina uma baseline e varia arquitetura, learning rate, batch size e regularização;
6. utiliza Adam, entropia cruzada e early stopping com restauração dos melhores pesos;
7. calcula acurácia, perda, gap entre treino e teste e principais confusões;
8. gera matrizes de confusão, curvas de aprendizado, tabelas e o relatório final.

O conjunto de teste só é utilizado na avaliação. A seleção durante o treinamento é feita com o conjunto de validação.

## Como instalar

É necessário Python 3.10 ou 3.11 (64 bits).

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Como executar

Para realizar todos os 24 experimentos pedidos na atividade:

```powershell
python mnist_mlp.py --mode full
```

No Windows também é possível usar o atalho:

```powershell
.\executar_completo.ps1
```

O primeiro uso baixa o MNIST automaticamente. A execução completa treina 24 configurações e pode demorar, especialmente sem GPU. Para validar o pipeline com uma amostra estratificada e menos combinações:

```powershell
python mnist_mlp.py --mode quick
```

No Windows, o equivalente é `.\executar_rapido.ps1`.

Também é possível limitar as épocas, por exemplo `python mnist_mlp.py --mode full --epochs 10`. O early stopping encerra antes quando a perda de validação deixa de melhorar.

### Opções da linha de comando

| Opção | Descrição |
|---|---|
| `--mode full` | Usa todo o MNIST e executa as 24 configurações. |
| `--mode quick` | Usa uma amostra estratificada e menos configurações para validação rápida. |
| `--epochs N` | Define o máximo de épocas. Sem essa opção, são 20 no modo full e 8 no quick. |

## Saída

Após a execução, abra `resultados/relatorio.md`. A pasta também conterá:

- `metricas.csv`: resultados finais de todos os testes;
- `metricas_parciais.csv`: checkpoint atualizado após cada modelo;
- `configuracoes.json`: hiperparâmetros exatos;
- `modelo_baseline.keras`: baseline salva;
- `figuras/`: comparativo, matrizes de confusão, curvas e erros.

O modo `full` deve ser usado para a entrega final; o relatório registra claramente qual modo gerou os resultados.
