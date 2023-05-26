from os import path
import sys
import os

diretorio_projeto_atual = path.dirname(__file__)
diretorio_projeto_atual = path.join("..")
sys.path.append(diretorio_projeto_atual)
diretorio_projeto_atual = os.path.join(os.path.dirname(__file__), "..")
diretorio_projeto_atual = os.path.normpath(diretorio_projeto_atual)
sys.path.append(diretorio_projeto_atual)


from src.gerenciadorDeArquivos.ler_arquivo import ler_arquivo
import numpy as np


def test_ler_arquivo():
    """
    Caso de teste: fornecendo um caminho de arquivo válido
    Caso de teste do src/arquivos/arq.dat e do src/arquivos/arq2.dat

    """
    caminho_arquivo = "arquivos/arq.dat"
    (
        quantidade_sistemas,
        ordem_da_matriz,
        precisao,
        matriz_A,
        vetores_b,
    ) = ler_arquivo(caminho_arquivo)
    assert quantidade_sistemas == 1
    assert ordem_da_matriz == 3
    assert precisao == 0.001
    assert np.array_equal(
        matriz_A,
        np.array([[5.0, 2.0, 1.0], [2.0, -1.0, 3.0], [0.0, 1.0, -2.0]]),
    )
    assert np.array_equal(vetores_b, [np.array([8.0, 9.0, -6.0])])

    caminho_arq2 = "arquivos/arq2.dat"

    (
        quantidade_sistemas,
        ordem_da_matriz,
        precisao,
        matriz_A,
        vetores_b_arq2,
    ) = ler_arquivo(caminho_arq2)

    assert quantidade_sistemas == 2
    assert ordem_da_matriz == 3
    assert precisao == 0.001
    assert np.array_equal(
        matriz_A, [[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]
    )
    assert np.array_equal(
        vetores_b_arq2, [np.array([6.0, -5.0, 3.0]), np.array([6.0, 7.0, -8.0])]
    )
