#!/usr/bin/env python3
"""
plota_curvas_igor.py
Script para leitura dos resultados de speedup e geração do gráfico comparativo
entre Speedup Ideal Linear, pi_mpi e Pipeline de PLN em Dask.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_grafico(speedup_pi_csv, speedup_corpus_csv, output_png):
    # Configuração de estilo visual do Seaborn/Matplotlib
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 14
    })

    # Carregamento dos dados dos CSVs
    df_pi = pd.read_csv(speedup_pi_csv)
    df_corpus = pd.read_csv(speedup_corpus_csv)

    # Tratamento pi_mpi: toma a menor medição t_total para cada número de processos (conforme metodologia da Aula 3)
    pi_summary = df_pi.groupby('nprocs')['t_total'].min().reset_index()
    t1_pi = pi_summary.loc[pi_summary['nprocs'] == 1, 't_total'].values[0]
    pi_summary['speedup'] = t1_pi / pi_summary['t_total']

    # Tratamento Corpus PLN: toma a mediana das medições t_total para cada configuração (conforme metodologia da Aula 5)
    corpus_summary = df_corpus.groupby('nprocs')['t_total'].median().reset_index()
    t1_corpus = corpus_summary.loc[corpus_summary['nprocs'] == 1, 't_total'].values[0]
    corpus_summary['speedup'] = t1_corpus / corpus_summary['t_total']

    workers = sorted(pi_summary['nprocs'].unique())

    # Criação da figura
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Curva 1: Speedup Ideal Linear (S = p)
    ax.plot(workers, workers, '--', color='#7f7f7f', label='Speedup Ideal Linear ($S = p$)', linewidth=2, alpha=0.8)

    # Curva 2: pi_mpi (CPU-Bound, C/MPI)
    ax.plot(pi_summary['nprocs'], pi_summary['speedup'], 'o-', color='#1f77b4', 
            label='pi_mpi (C / OpenMPI - CPU Bound)', linewidth=2.5, markersize=8)

    # Curva 3: Pipeline Corpus PLN (I/O & Network-Bound, Dask/Python)
    ax.plot(corpus_summary['nprocs'], corpus_summary['speedup'], 's-', color='#d62728', 
            label='Corpus TF-IDF (Dask / Python - I/O & Shuffle Bound)', linewidth=2.5, markersize=8)

    # Ajuste dos eixos (escala log2 no eixo X para destacar os pontos 1, 2, 4, 8, 16, 32)
    ax.set_xscale('log', base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Anotações dos pontos extremos de speedup
    s_pi_max = pi_summary.loc[pi_summary['nprocs'] == 32, 'speedup'].values[0]
    s_corp_max = corpus_summary.loc[corpus_summary['nprocs'] == 32, 'speedup'].values[0]
    
    ax.annotate(f'{s_pi_max:.2f}x', (32, s_pi_max), textcoords="offset points", xytext=(-25, 10), ha='center',
                fontweight='bold', color='#1f77b4', fontsize=10)
    ax.annotate(f'{s_corp_max:.2f}x', (32, s_corp_max), textcoords="offset points", xytext=(0, -18), ha='center',
                fontweight='bold', color='#d62728', fontsize=10)

    # Títulos e rótulos
    ax.set_xlabel('Número de Workers / Processos ($p$)', fontweight='bold')
    ax.set_ylabel('Speedup $S(p) = T(1) / T(p)$', fontweight='bold')
    ax.set_title('Análise Comparativa de Speedup: pi_mpi vs. Pipeline de PLN (Dask)', fontweight='bold', pad=15)

    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Gráfico gerado com sucesso em: {output_png}")

if __name__ == '__main__':
    gerar_grafico('resultados/speedup.csv', 'resultados/speedup_corpus.csv', 'resultados/curvas_comparativas_igor.png')
