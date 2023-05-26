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

from src.metodos.gaussSeidelClass import MetodoGaussSeidel


def test_gauss_seidel_sistlin():
    seide = MetodoGaussSeidel()

    seide.calcula_sistema_linear(
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

    assert seide.get_resultados()[0][0] == 0.5006184545345604
    assert seide.get_resultados()[0][1] == 0.5031799676362425
    assert seide.get_resultados()[0][2] == 0.5002502287970856
    assert seide.get_resultados()[0][3] == 0.49797567451605573


test_gauss_seidel_sistlin()


def test_gauss_seidel_nao_converge_com_converge():
    """
    Fornece o exercicio numero 11.5 da lista de exercicios
    """
    seidel = MetodoGaussSeidel()

    for j in range(0, 2):
        seidel.calcula_sistema_linear(
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
        if seidel.get_converge() == True:
            break

    assert seidel.get_converge() == False
    seidel.set_converge(True)

    for j in range(0, 2):
        seidel.calcula_sistema_linear(
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
            0.001,
        )
        if seidel.get_converge() == True:
            break

    assert seidel.get_resultados()[1][0] == 2.039674994794027
    assert seidel.get_resultados()[1][1] == 5.874032938835107
    assert seidel.get_resultados()[1][2] == -0.9809091904867938
    assert seidel.get_resultados()[1][3] == 0.9519832812664585


def test_gauss_seidel_exerc_11_5_lista():
    """
    Fornece o exercicio numero 11.5 da lista de exercicios
    """
    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
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
        0.1,
    )

    assert seidel.get_converge() == False


def test_seidel_exerc_11_9_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.9
    """

    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
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

    assert seidel.get_resultados()[0][0] == 0.36363351717591286
    assert seidel.get_resultados()[0][1] == 0.45454359147697687
    assert seidel.get_resultados()[0][2] == 0.4545447009149939
    assert seidel.get_resultados()[0][3] == 0.36363617522874847


def test_seidel_exerc_11_8_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.8
    """
    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
        3,
        np.array([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]]),
        np.array([12.0, 12.0, 12.0]),
        0.00010000,
    )

    assert seidel.get_resultados()[0][0] == 1.0000009946400001
    assert seidel.get_resultados()[0][1] == 1.0000005695760001
    assert seidel.get_resultados()[0][2] == 0.9999998435784001


def test_seidel_arq2():
    #     """
    #     Fornecendo os valores individualmente do  arquivo2.txt
    #     e verificando o resultado das variaveis.
    #     """

    seidel = MetodoGaussSeidel()
    for j in range(0, 2):
        seidel.calcula_sistema_linear(
            3,
            np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
            np.array([6.0, -5.0, 3.0]),
            0.001,
        )
        if seidel.get_converge() == True:
            break

    assert seidel.get_resultados()[0][0] == 2.0001325794897116
    assert seidel.get_resultados()[0][1] == 0.0004496503827800424
    assert seidel.get_resultados()[0][2] == 0.9998328125678579

    for j in range(0, 2):
        seidel.calcula_sistema_linear(
            3,
            np.array([[1.0, 3.0, 4.0], [-2.0, 5.0, -1.0], [3.0, -2.0, -3.0]]),
            np.array([6.0, 7.0, -8.0]),
            0.001,
        )
        if seidel.get_converge() == True:
            break

    assert seidel.get_resultados()[1][0] == -1.0106679402542502
    assert seidel.get_resultados()[1][1] == 1.1705474277911718
    assert seidel.get_resultados()[1][2] == 0.8756337745516352


def test_seidel_arq1():
    """
    Fornece como entrada o sistema do primeiro arquivo: arq.txt
    """

    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
        3,
        np.array([[5.0, 2.0, 1.0], [2.0, -1.0, 3.0], [0.0, 1.0, -2.0]]),
        np.array([8.0, 9.0, -6.0]),
        0.001,
    )

    assert seidel.get_resultados()[0][0] == -1.0025390624999992
    assert seidel.get_resultados()[0][1] == 4.0025390625
    assert seidel.get_resultados()[0][2] == 5.00126953125


def test_gauss_seidel_exerc_11_7_lista():
    """
    Fornece o exercicio numero 11.7 da lista de exercicios
    """
    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
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
        0.001,
    )

    assert seidel.get_resultados()[0][0] == 2.039674994794027
    assert seidel.get_resultados()[0][1] == 5.874032938835107
    assert seidel.get_resultados()[0][2] == -0.9809091904867938
    assert seidel.get_resultados()[0][3] == 0.9519832812664585


def test_gauss_seidel_quantidade_sistemas():
    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
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

    assert seidel.get_resultados()[0][0] == 0.36363351717591286
    assert seidel.get_resultados()[0][1] == 0.45454359147697687
    assert seidel.get_resultados()[0][2] == 0.4545447009149939
    assert seidel.get_resultados()[0][3] == 0.36363617522874847

    seidel.set_converge(True)

    seidel.calcula_sistema_linear(
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
        0.001,
    )

    assert seidel.get_resultados()[1][0] == 2.039674994794027
    assert seidel.get_resultados()[1][1] == 5.874032938835107
    assert seidel.get_resultados()[1][2] == -0.9809091904867938
    assert seidel.get_resultados()[1][3] == 0.9519832812664585

    seidel.calcula_sistema_linear(
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

    assert seidel.get_resultados()[2][0] == 0.8713671149422959
    assert seidel.get_resultados()[2][1] == 0.7202775862647929
    assert seidel.get_resultados()[2][2] == -0.9871370134519344
    assert seidel.get_resultados()[2][3] == 0.2025760553584851


def test_seidel_exerc_11_6_lista():
    """
    Fornece como entrada o sistema da lista de exericio numero 11.6
    """
    seidel = MetodoGaussSeidel()

    seidel.calcula_sistema_linear(
        3,
        np.array([[9.0, -6.0, 3.0], [-6.0, 29.0, -7.0], [3.0, -7.0, 18.0]]),
        np.array([-3.0, -8.0, 33.0]),
        0.001,
    )

    assert seidel.get_resultados()[0][0] == -1.0000181708460845
    assert seidel.get_resultados()[0][1] == 3.644054192883096e-05
    assert seidel.get_resultados()[0][2] == 2.0000171997962086
