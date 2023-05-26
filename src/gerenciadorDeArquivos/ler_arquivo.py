import numpy as np
from typing import Tuple, List


def ler_arquivo(
    caminho_arquivo: str,
) -> Tuple[int, int, float, List[List[float]], List[List[float]]]:
    """
    Verificando se arquivo existe e caso exista os dados do arquivo sao armazenados
    nas suas respectivas variáveis.

    :param - caminho_arquivo: caminho do arquivo que será verificado.
    :return - Retorna a quantidade de sistemas do arquivo, a ordem da matriz,
              a precisão e o sistema linear a ser resolvido.
    """

    np.set_printoptions(precision=6)

    quantidade_sistemas = np.loadtxt(
        fname=caminho_arquivo, dtype=int, delimiter=" ", max_rows=1, usecols=(0)
    )
    ordem_da_matriz = np.loadtxt(
        fname=caminho_arquivo, dtype=int, delimiter=" ", max_rows=1, usecols=(1)
    )
    precisao = np.loadtxt(
        fname=caminho_arquivo,
        dtype=np.float64,
        delimiter=" ",
        max_rows=1,
        usecols=(2),
    )

    matriz_A = np.loadtxt(
        fname=caminho_arquivo,
        dtype=np.longdouble,
        delimiter=" ",
        max_rows=ordem_da_matriz,
        skiprows=1,
        usecols=np.arange(0, ordem_da_matriz),
    )
    vetores_b = []

    if quantidade_sistemas == 1:
        for i in range(1, quantidade_sistemas + 1):
            vetor_b = np.loadtxt(
                fname=caminho_arquivo,
                dtype=np.longdouble,
                delimiter=" ",
                max_rows=ordem_da_matriz,
                skiprows=ordem_da_matriz + i,
            )
            vetores_b.append(vetor_b)
    else:
        for i in range(1, quantidade_sistemas + 1):
            vetor_b = np.loadtxt(
                fname=caminho_arquivo,
                dtype=np.longdouble,
                delimiter=" ",
                max_rows=1,
                skiprows=ordem_da_matriz + i,
            )
            vetores_b.append(vetor_b)

    return quantidade_sistemas, ordem_da_matriz, precisao, matriz_A, vetores_b
