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

from src.metodos.gaussClasse import MetodoGauss


def test_gauss():
    """
    Fornecendo o arquivo3.txt para testar o pivoteamento e verificar o resultado das variaveis
    """

    gauss = MetodoGauss()

    # teste do arquivo3.txt
    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[0.0, 1.0, -2.0], [2.0, -1.0, 3.0], [5.0, 2.0, 1.0]]),
        np.array([-6.0, 9.0, 8.0]),
    )
    assert gauss.get_resultados()[0][0] == -1.0000000000000013
    assert gauss.get_resultados()[0][1] == 4.000000000000003
    assert gauss.get_resultados()[0][2] == 5.000000000000002


def test_gauss_exerc_video():
    """
    Esse sistema foi retirado do vídeo do professor resolvendo um sistema 4x4
    """
    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [1.0, 6.0, 2.0, 4.0],
                [3.0, 19.0, 4.0, 15.0],
                [1.0, 4.0, 8.0, -12.0],
                [5.0, 33.0, 9.0, 3.0],
            ]
        ),
        np.array([8.0, 25.0, 18.0, 72.0]),
    )

    assert gauss.get_resultados()[0][0] == -137.99999999999312
    assert gauss.get_resultados()[0][1] == 19.99999999999913
    assert gauss.get_resultados()[0][2] == 10.999999999999405
    assert gauss.get_resultados()[0][3] == 0.9999999999998854


def test_gauss_exerc_11_1_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.1
    """

    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[1.0, 2.0, 4.0], [-3.0, -1.0, 4.0], [2.0, 14.0, 5.0]]),
        np.array([13.0, 8.0, 50.0]),
    )
    assert gauss.get_resultados()[0][0] == -1.0
    assert gauss.get_resultados()[0][1] == 3.0
    assert gauss.get_resultados()[0][2] == 2.0


def test_gauss_exer_11_3_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero  11.3
    """

    gauss = MetodoGauss()
    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [1.0, -3.0, 5.0, 6.0],
                [-8.0, 4.0, -1.0, 0.0],
                [3.0, 2.0, -2.0, 7.0],
                [1.0, 2.0, 5.0, -4.0],
            ]
        ),
        np.array([17.0, 29.0, -11.0, 7.0]),
    )

    assert gauss.get_resultados()[0][0] == -4.0
    assert gauss.get_resultados()[0][1] == 2.5376526277146434e-16
    assert gauss.get_resultados()[0][2] == 2.9999999999999996
    assert gauss.get_resultados()[0][3] == 0.9999999999999998


def test_gauss_exer_11_7_lista():
    """
    Fornece o exercicio numero 11.7 da lista de exercicios
    """
    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        np.array([2.0, 2.0, -1.0, -2.0]),
    )

    assert gauss.get_resultados()[0][0] == 1.999999999999996
    assert gauss.get_resultados()[0][1] == 6.0000000000000115
    assert gauss.get_resultados()[0][2] == -1.0000000000000016
    assert gauss.get_resultados()[0][3] == 1.0000000000000044


def test_gauss_quantidade_sistemas():
    """
    Testando se ao passar 3 sistemas o resultado e o correto.
    """

    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        np.array([2.0, 2.0, -1.0, -2.0]),
    )

    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        np.array([2.0, -2.0, -1.0, -2.0]),
    )

    gauss.calcula_matriz_triangular_inferior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        np.array([-2.0, -2.0, -1.0, -2.0]),
    )

    assert gauss.get_resultados()[0][0] == 1.999999999999996
    assert gauss.get_resultados()[0][1] == 6.0000000000000115
    assert gauss.get_resultados()[0][2] == -1.0000000000000016
    assert gauss.get_resultados()[0][3] == 1.0000000000000044

    assert gauss.get_resultados()[1][0] == 8.666666666666675
    assert gauss.get_resultados()[1][1] == -20.222222222222246
    assert gauss.get_resultados()[1][2] == 2.5555555555555585
    assert gauss.get_resultados()[1][3] == -8.333333333333343

    assert gauss.get_resultados()[2][0] == 1.6666666666666743
    assert gauss.get_resultados()[2][1] == -13.555555555555571
    assert gauss.get_resultados()[2][2] == 1.888888888888891
    assert gauss.get_resultados()[2][3] == -4.33333333333334


def test_gauss_sistema_2x2():
    """
    Sistema 2x2
    """
    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        2, np.array([[4.0, -2.0], [-2.0, 2.0]]), np.array([-2.0, -2.0])
    )

    assert gauss.get_resultados()[0][0] == -2
    assert gauss.get_resultados()[0][1] == -3


def test_gauss_exerc_11_2_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.2
    """

    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[-2.0, 3.0, 1.0], [2.0, 1.0, -4.0], [4.0, 10.0, -6.0]]),
        np.array([-5.0, -9.0, 2.0]),
    )

    assert gauss.get_resultados()[0][0] == 7.0
    assert gauss.get_resultados()[0][1] == 1.0
    assert gauss.get_resultados()[0][2] == 6.0


def test_gauss_exerc_11_4_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.4
    """

    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[-5.0, -1.0, 4.0], [2.0, 4.0, 1.0], [1.0, 2.0, 3.0]]),
        np.array([-2.0, 24.0, 17.0]),
    )

    assert gauss.get_resultados()[0][0] == 1.0000000000000009
    assert gauss.get_resultados()[0][1] == 4.999999999999999
    assert gauss.get_resultados()[0][2] == 2.000000000000001


def test_gauss_arq1():
    """
    Fornecendo o arq.txt para testar o pivoteamento e verificar o resultado das variaveis
    """
    gauss = MetodoGauss()

    # teste do arq.txt
    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[5.0, 2.0, 1.0], [2.0, -1.0, 3.0], [0.0, 1.0, -2.0]]),
        np.array([8.0, 9.0, -6.0]),
    )
    assert gauss.get_resultados()[0][0] == -1.0000000000000013
    assert gauss.get_resultados()[0][1] == 4.000000000000003
    assert gauss.get_resultados()[0][2] == 5.000000000000002


def test_gauss_arq2():
    """
    Fornecendo os valores individualmente do  arquivo2.txt
    e verificando o resultado das variaveis.
    """

    gauss = MetodoGauss()

    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
        np.array([6.0, -5.0, 3.0]),
    )

    assert gauss.get_resultados()[0][0] == 2.0
    assert gauss.get_resultados()[0][1] == 0.0
    assert gauss.get_resultados()[0][2] == 1.0

    # segundo sistema do arquivo2.txt
    gauss.calcula_matriz_triangular_inferior(
        3,
        np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
        np.array([6.0, 7.0, -8.0]),
    )

    assert gauss.get_resultados()[1][0] == -1.0113636363636367
    assert gauss.get_resultados()[1][1] == 1.1704545454545452
    assert gauss.get_resultados()[1][2] == 0.8749999999999999
