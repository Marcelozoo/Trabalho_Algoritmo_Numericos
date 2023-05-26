from os import path
import sys
import os
import numpy as np

diretorio_projeto_atual = path.dirname(__file__)
diretorio_projeto_atual = path.join("..")
sys.path.append(diretorio_projeto_atual)
diretorio_projeto_atual = os.path.join(os.path.dirname(__file__), "..")
diretorio_projeto_atual = os.path.normpath(diretorio_projeto_atual)
sys.path.append(diretorio_projeto_atual)

from src.metodos.gaussJacobiClass import MetodoGaussJacobi


def test_gauss_jacobi_sistlin():
    """
    Fornece os dados do arquivo sistlin3x4x4.dat
    """
    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [2.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 2.0],
            ]
        ),
        np.array([2.5, 2.5, 2.5, 2.5]),
        0.010000,
    )

    assert jacobi.get_converge() == False


def test_gauss_jacobi_exerc_11_5_lista():
    """
    Fornece o exercicio numero 11.5 da lista de exercicios
    """
    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [5.0, 5.0, 1.0, 0.0],
                [1.0, 6.0, 2.0, -3.0],
                [2.0, 4.0, -2.0, 1.0],
                [4.0, -1.0, 3.0, 8.0],
            ]
        ),
        np.array([18.0, 7.0, 8.0, 43.0]),
        0.001,
    )
    assert jacobi.get_converge() == False


def test_jacobi_arq2():
    """
    Fornecendo os valores individualmente do  arquivo2.dat
    e verificando o resultado das variaveis.
    """

    jacobi = MetodoGaussJacobi()
    for j in range(0, 2):
        jacobi.calcula_sistema_linear(
            3,
            np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
            np.array([6.0, -5.0, 3.0]),
            0.001,
        )
        if jacobi.get_converge() == True:
            break

    assert jacobi.get_resultados()[0][0] == 1.9986968609625804
    assert jacobi.get_resultados()[0][1] == -0.0006392434427507255
    assert jacobi.get_resultados()[0][2] == 1.0000437409710923

    #     # segundo sistema do arquivo2.txt
    for j in range(0, 2):
        jacobi.calcula_sistema_linear(
            3,
            np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
            np.array([6.0, 7.0, -8.0]),
            0.001,
        )

        if jacobi.get_converge() == True:
            break

    assert jacobi.get_resultados()[1][0] == -1.0117794756802088
    assert jacobi.get_resultados()[1][1] == 1.1704481257599657
    assert jacobi.get_resultados()[1][2] == 0.8743779149611439


def test_jacobi_arq1():
    """
    Fornece como entrada o sistema do primeiro arquivo: arq.dat
    """

    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        3,
        np.array([[5.0, 2.0, 1.0], [2.0, -1.0, 3.0], [0.0, 1.0, -2.0]]),
        np.array([8.0, 9.0, -6.0]),
        0.001,
    )

    assert jacobi.get_resultados()[0][0] == -1.000732885615329
    assert jacobi.get_resultados()[0][1] == 3.9976317087455637
    assert jacobi.get_resultados()[0][2] == 5.001241531749206


def test_jacobi_exerc_11_8_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.8
    """
    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        3,
        np.array([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]]),
        np.array([12.0, 12.0, 12.0]),
        0.00010000,
    )

    assert jacobi.get_resultados()[0][0] == 1.0000128
    assert jacobi.get_resultados()[0][1] == 1.0000128
    assert jacobi.get_resultados()[0][2] == 1.0000128


def test_jacobi_exerc_11_9_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.9
    """

    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [4.0, -1.0, 0.0, 0.0],
                [-1.0, 4.0, -1.0, 0.0],
                [0.0, -1.0, 4.0, -1.0],
                [0.0, 0.0, -1.0, 4.0],
            ]
        ),
        np.array([1.0, 1.0, 1.0, 1.0]),
        0.00010000,
    )

    assert jacobi.get_resultados()[0][0] == 0.36362195014953613
    assert jacobi.get_resultados()[0][1] == 0.45452213287353516
    assert jacobi.get_resultados()[0][2] == 0.45452213287353516
    assert jacobi.get_resultados()[0][3] == 0.36362195014953613


def test_jacobi_exerc_video():
    """
    Testando com o sistema que foi resolvido pelo professor em vídeo.

    """
    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [5.0, 2.0, 0.0, 1.0],
                [1.0, 8.0, -3.0, 2.0],
                [0.0, 1.0, 6.0, 1.0],
                [1.0, -1.0, 2.0, 9.0],
            ]
        ),
        np.array([6.0, 10.0, -5.0, 0.0]),
        0.0010000,
    )

    assert jacobi.get_resultados()[0][0] == 0.8717241422245721
    assert jacobi.get_resultados()[0][1] == 0.7204416662548662
    assert jacobi.get_resultados()[0][2] == -0.9869275092408688
    assert jacobi.get_resultados()[0][3] == 0.20238329564614765


def test_gauss_jacobi_exerc_11_4_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.4
    """

    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        3,
        np.array([[-5.0, -1.0, 4.0], [2.0, 4.0, 1.0], [1.0, 2.0, 3.0]]),
        np.array([-2.0, 24.0, 17.0]),
        0.001,
    )

    assert jacobi.get_resultados()[0][0] == 1.0026692708333331
    assert jacobi.get_resultados()[0][1] == 4.999397786458333
    assert jacobi.get_resultados()[0][2] == 1.9995442708333329


def test_gauss_jacobi_quantidade_sistemas():
    """
    Testa se calcula o resultado correto quando mais de um sistema é passado.
    """

    jacobi = MetodoGaussJacobi()

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [4.0, -1.0, 0.0, 0.0],
                [-1.0, 4.0, -1.0, 0.0],
                [0.0, -1.0, 4.0, -1.0],
                [0.0, 0.0, -1.0, 4.0],
            ]
        ),
        np.array([1.0, 1.0, 1.0, 1.0]),
        0.00010000,
    )

    assert jacobi.get_resultados()[0][0] == 0.36362195014953613
    assert jacobi.get_resultados()[0][1] == 0.45452213287353516
    assert jacobi.get_resultados()[0][2] == 0.45452213287353516
    assert jacobi.get_resultados()[0][3] == 0.36362195014953613

    jacobi.calcula_sistema_linear(
        4,
        np.array(
            [
                [5.0, 2.0, 0.0, 1.0],
                [1.0, 8.0, -3.0, 2.0],
                [0.0, 1.0, 6.0, 1.0],
                [1.0, -1.0, 2.0, 9.0],
            ]
        ),
        np.array([6.0, 10.0, -5.0, 0.0]),
        0.0010000,
    )

    assert jacobi.get_resultados()[1][0] == 0.8717241422245721
    assert jacobi.get_resultados()[1][1] == 0.7204416662548662
    assert jacobi.get_resultados()[1][2] == -0.9869275092408688
    assert jacobi.get_resultados()[1][3] == 0.20238329564614765
