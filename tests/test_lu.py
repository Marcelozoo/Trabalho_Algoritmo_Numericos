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

from src.metodos.LuClasse import MetodoLu
from scipy.linalg import lu_factor
from scipy.linalg import lu_factor, lu_solve


def test_lu_arq1():
    """
    Fornecendo o arq.txt para testar o pivoteamento e verificar o resultado das variaveis
    """
    lu = MetodoLu()

    # teste do arq.txt
    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[5.0, 2.0, 1.0], [2.0, -1.0, 3.0], [0.0, 1.0, -2.0]]),
        np.array([8.0, 9.0, -6.0]),
        False,
    )

    assert lu.get_resultados()[0][0] == -1.0000000000000013
    assert lu.get_resultados()[0][1] == 4.000000000000003
    assert lu.get_resultados()[0][2] == 5.000000000000002


def test_lu_arq2():
    """
    Fornecendo os valores individualmente do  arquivo2.txt
    e verificando o resultado das variaveis.
    """

    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
        np.array([6.0, -5.0, 3.0]),
        False,
    )

    assert lu.get_resultados()[0][0] == 2.0
    assert lu.get_resultados()[0][1] == 0.0
    assert lu.get_resultados()[0][2] == 1.0

    # segundo sistema do arquivo2.txt
    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
        np.array([6.0, 7.0, -8.0]),
        False,
    )

    assert lu.get_resultados()[1][0] == -1.0113636363636365
    assert lu.get_resultados()[1][1] == 1.1704545454545454
    assert lu.get_resultados()[1][2] == 0.875


def test_lu_quantidade_sistemas():
    """
    Testando se ao passar 3 sistemas o resultado e o correto.
    """

    lu = MetodoLu()
    matriz_vetores_b = np.array(
        [
            [
                [2.0, 2.0, -1.0, -2.0],
                [2.0, -2.0, -1.0, -2.0],
                [-2.0, -2.0, -1.0, -2.0],
            ]
        ]
    )

    print(matriz_vetores_b[0][0])
    lu.calcula_matriz_triangular_inferior_e_superior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        matriz_vetores_b[0][0],
        False,
    )

    lu.calcula_matriz_triangular_inferior_e_superior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        matriz_vetores_b[0][1],
        True,
    )

    lu.calcula_matriz_triangular_inferior_e_superior(
        4,
        np.array(
            [
                [4.0, -2.0, 4.0, 10.0],
                [-2.0, 2.0, -1.0, -7.0],
                [4.0, -1.0, 14.0, 11.0],
                [10.0, -7.0, 11.0, 31.0],
            ]
        ),
        matriz_vetores_b[0][2],
        True,
    )

    assert lu.get_resultados()[0][0] == 1.999999999999996
    assert lu.get_resultados()[0][1] == 6.00000000000001
    assert lu.get_resultados()[0][2] == -1.0000000000000013
    assert lu.get_resultados()[0][3] == 1.000000000000004

    assert lu.get_resultados()[1][0] == 8.666666666666675
    assert lu.get_resultados()[1][1] == -20.222222222222246
    assert lu.get_resultados()[1][2] == 2.5555555555555585
    assert lu.get_resultados()[1][3] == -8.333333333333343

    assert lu.get_resultados()[2][0] == 1.6666666666666743
    assert lu.get_resultados()[2][1] == -13.555555555555571
    assert lu.get_resultados()[2][2] == 1.888888888888891
    assert lu.get_resultados()[2][3] == -4.33333333333334


def test_lu_exerc_11_7_lista():
    """
    Fornece o exercicio numero 11.7 da lista de exercicios
    """
    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
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
        False,
    )

    assert lu.get_resultados()[0][0] == 1.999999999999996
    assert lu.get_resultados()[0][1] == 6.00000000000001
    assert lu.get_resultados()[0][2] == -1.0000000000000013
    assert lu.get_resultados()[0][3] == 1.000000000000004


def test_lu_exerc_video():
    """
    Esse sistema foi retirado do vídeo do professor resolvendo um sistema 4x4
    """
    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
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
        False,
    )

    assert lu.get_resultados()[0][0] == -137.9999999999909
    assert lu.get_resultados()[0][1] == 19.999999999998852
    assert lu.get_resultados()[0][2] == 10.99999999999921
    assert lu.get_resultados()[0][3] == 0.9999999999998472


def test_lu_arq3():
    """
    Fornecendo o arquivo3.txt para testar o pivoteamento e verificar o resultado das variaveis
    """

    lu = MetodoLu()

    # teste do arquivo3.txt
    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[0.0, 1.0, -2.0], [2.0, -1.0, 3.0], [5.0, 2.0, 1.0]]),
        np.array([-6.0, 9.0, 8.0]),
        False,
    )

    assert lu.get_resultados()[0][0] == -1.0000000000000013
    assert lu.get_resultados()[0][1] == 4.000000000000003
    assert lu.get_resultados()[0][2] == 5.000000000000002


def test_lu_exerc_11_1_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.1
    """

    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[1.0, 2.0, 4.0], [-3.0, -1.0, 4.0], [2.0, 14.0, 5.0]]),
        np.array([13.0, 8.0, 50.0]),
        False,
    )
    assert lu.get_resultados()[0][0] == -1.0
    assert lu.get_resultados()[0][1] == 3.0
    assert lu.get_resultados()[0][2] == 2.0


def test_lu_exer_11_3_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero  11.3
    """

    lu = MetodoLu()
    lu.calcula_matriz_triangular_inferior_e_superior(
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
        False,
    )

    assert lu.get_resultados()[0][0] == -3.9999999999999996
    assert lu.get_resultados()[0][1] == 7.61295788314393e-16
    assert lu.get_resultados()[0][2] == 3.0000000000000004
    assert lu.get_resultados()[0][3] == 0.9999999999999998


def test_lu_sistema_2x2():
    """
    Sistema 2x2
    """
    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
        2, np.array([[4.0, -2.0], [-2.0, 2.0]]), np.array([-2.0, -2.0]), False
    )

    assert lu.get_resultados()[0][0] == -2
    assert lu.get_resultados()[0][1] == -3


def test_resultado4():
    lu = MetodoLu()
    lu.calcula_matriz_triangular_inferior_e_superior(
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
        False,
    )

    assert lu.get_resultados()[0][0] == 8.666666666666675
    assert lu.get_resultados()[0][1] == -20.222222222222246
    assert lu.get_resultados()[0][2] == 2.5555555555555585
    assert lu.get_resultados()[0][3] == -8.333333333333343


def test_lu_exerc_11_6_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.6
    """

    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
        3,
        np.array([[9.0, -6.0, 3.0], [-6.0, 29.0, -7.0], [3.0, -7.0, 18.0]]),
        np.array([-3.0, -8.0, 33.0]),
        False,
    )
    assert lu.get_resultados()[0][0] == -1.0
    assert lu.get_resultados()[0][1] == 0.0
    assert lu.get_resultados()[0][2] == 2.0


def test_lu_exerc_11_5_lista():
    """
    Fornece o exercicio numero 11.5 da lista de exercicios
    """
    lu = MetodoLu()

    lu.calcula_matriz_triangular_inferior_e_superior(
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
        False,
    )

    assert lu.get_resultados()[0][0] == 1.0000000000000007
    assert lu.get_resultados()[0][1] == 1.9999999999999993
    assert lu.get_resultados()[0][2] == 3.000000000000001
    assert lu.get_resultados()[0][3] == 3.9999999999999996
