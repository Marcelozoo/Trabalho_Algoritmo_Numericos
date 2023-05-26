from typing import List


import warnings
import math
import numpy as np


class MetodoGaussSeidel:

    """
    Essa classe realiza todos os métodos necessáiros para resolver o sistema através
    do método de Gauss-Seidel.
    """

    def __init__(self) -> None:
        self.precisao_obtida = 0
        self.__precisao = []
        self.__resultados = []
        self.__resultados_tempo = []
        self.__converge = True
        self.__colunas_invertidas = []

    def calcula_sistema_linear(
        self,
        ordem_matriz: int,
        matriz_A: List[List[float]],
        vetor_b: List[float],
        precisao: float,
    ) -> None:
        """
        Essa função realiza os calculos para descobrir o primeiro vetor x, o vetor x0.

        :param
           - ordem_matriz: a ordem da matriz.
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - vetor_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
           - precisão: precisão que deve ser alcançada para descobrir a resposta das variáveis.
        :return - vazio

        """

        vetor_x = []

        if self.__converge == False:
            self.__pega_pivo(matriz_A, ordem_matriz, vetor_b)

        for i in range(0, ordem_matriz):
            valor_x = vetor_b[i] / matriz_A[i][i]

            vetor_x.append(valor_x)

        vetor_x_novo = self.__realiza_substituicao(
            matriz_A, ordem_matriz, vetor_b, vetor_x, precisao
        )

        while self.precisao_obtida > precisao:
            vetor_x_novo = self.__realiza_substituicao(
                matriz_A, ordem_matriz, vetor_b, vetor_x_novo, precisao
            )

        self.precisao_obtida = 0

    def __realiza_substituicao(
        self,
        matriz_A: List[List[float]],
        ordem_matriz: int,
        vetor_b: List[float],
        vetor_x: List[float],
        precisao: float,
    ) -> List[float]:
        """
        Essa função realiza a substituição para descobrir o novo vetor , o vetor x1.
        :param
            - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - ordem_matriz: a ordem da matriz.
           - vetor_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
           - vetor_x: vetor que representa o vetor x0 inicialmente, mas ele pode mudar para x1, x2, etc.
           - precisão: precisão que deve ser alcançada para descobrir a resposta das variáveis.
        :return - vazio

        """

        vetor_x_novo = []
        pivos = 0
        soma = 0

        copia_vetor_x = vetor_x.copy()

        for i in range(0, ordem_matriz):
            for j in range(0, ordem_matriz):
                if i == j:
                    pivos = matriz_A[i][j]
                else:
                    with warnings.catch_warnings(record=True) as w:
                        soma = soma + (-matriz_A[i][j]) * vetor_x[j]

            with warnings.catch_warnings(record=True) as w:
                valor_x_novo = (vetor_b[i] + soma) / (pivos)
                vetor_x[i] = valor_x_novo
            vetor_x_novo.append(valor_x_novo)
            soma = 0

        return self.__criterio_de_parada(vetor_x_novo, copia_vetor_x, precisao)

    def __criterio_de_parada(
        self, vetor_x_novo: List[float], vetor_x: List[float], precisao: float
    ) -> None:
        """
        Essa função calcula a precisão alcançada pelo metodo  e verifica se a precisão alcançada é
        maior ou menor que a precisão desejada.

        :param
           - vetor_x_novo: vetor com os novos valores de x, representado o que seria o x1.
           - vetor_x: vetor que representa o vetor x0 inicialmente, mas ele pode mudar para x1, x2, etc.
           - precisão: precisão que deve ser alcançada para descobrir a resposta das variáveis.

        :return - vazio

        """
        valores = []
        for i in range(len(vetor_x)):
            with warnings.catch_warnings(record=True) as w:
                valores.append((vetor_x[i] - vetor_x_novo[i]))

        maior_valor_absoluto_x_novo = max(abs(num) for num in valores)
        maior_valor_absoluto_x = max(abs(num) for num in vetor_x_novo)

        with warnings.catch_warnings(record=True) as w:
            self.precisao_obtida = (
                maior_valor_absoluto_x_novo / maior_valor_absoluto_x
            )

        if self.precisao_obtida > precisao:
            return vetor_x_novo
        elif math.isnan(self.precisao_obtida):
            self.__converge = False
            self.__resultados.append(vetor_x_novo)
            self.__precisao.append(self.precisao_obtida)
            self.__colunas_invertidas.clear()
        else:
            if len(self.__colunas_invertidas) != 0:
                self.__corrigi_resultado(vetor_x_novo)
            tem_nan = False
            for i in range(0, len(self.__precisao)):
                if math.isnan(self.__precisao[i]):
                    tem_nan = True

            if tem_nan == True:
                self.__resultados[i] = vetor_x_novo
                self.__precisao[i] = self.precisao_obtida
                self.__converge = True
                self.__colunas_invertidas.clear()

            else:
                self.__resultados.append(vetor_x_novo)
                self.__precisao.append(self.precisao_obtida)
                self.__colunas_invertidas.clear()
                self.__converge = True

    def __corrigi_resultado(self, vetor_x_novo) -> None:
        """
        Essa função corrigi o vetor resultado caso tenha ocorrido algum pivoteamento.

        :param
           - vetor_x_novo: vetor com os resultados das variaveis x,y,z,etc...
        :return - vazio

        """
        if len(self.__colunas_invertidas) != 0:
            for i in range(0, len(self.__colunas_invertidas), 2):
                (
                    vetor_x_novo[self.__colunas_invertidas[i]],
                    vetor_x_novo[self.__colunas_invertidas[i + 1]],
                ) = (
                    vetor_x_novo[self.__colunas_invertidas[i + 1]],
                    vetor_x_novo[self.__colunas_invertidas[i]],
                )

    def get_converge(self) -> bool:
        """
        Essa função serve para alterar o valor de convergência depois que um sistema ja foi testado

        :return - vazio

        """

        return self.__converge

    def set_converge(self, converge) -> None:
        """
        Essa função serve para alterar o valor de convergência depois que um sistema ja foi testado

        :return - vazio

        """
        self.__converge = converge

    def imprimir_resultados(self, qtd_sistemas: int) -> None:
        """
        Apenas imprimi os resultados obtidos pelo método gauss-seidel.

        :return - vazio

        """
        print("Resultados dos sistemas resolvidos pelo Metodo de Gauss-Seidel:")
        if math.isnan(self.__precisao[qtd_sistemas]):
            print(f"Resultado para o sistema {qtd_sistemas+1}: Diverge!")

        else:
            print(
                f"Resultado para o sitema {qtd_sistemas+1}: {self.__resultados[qtd_sistemas]} Tempo: {self.__resultados_tempo[qtd_sistemas]} segundos"
            )

    def add_resultados_tempo(self, tempo: float) -> None:
        """
        Esse método serve para adicionar o tempo gasto para calcular um dado sistema .

        :return - vazio

        """
        self.__resultados_tempo.append(tempo)

    def __pega_pivo(self, matriz_A, ordem_matriz, vetor_b) -> None:
        """
        Essa função percorre a matriz_A e passa os pivos para a função self.__pivoteai_matriz_A.

        :param
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - ordem_matriz: a ordem da matriz.
           - vetor_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """
        for i in range(0, ordem_matriz):
            for j in range(0, ordem_matriz):
                if i == j:
                    pivo = matriz_A[i][j]
                    self.__pivoteai_matriz_A(
                        matriz_A, ordem_matriz, pivo, i, j, vetor_b
                    )

    def __pivoteai_matriz_A(
        self, matriz_A, ordem_matriz, pivo, pivo_linha, pivo_coluna, vetor_b
    ) -> None:
        """
        Essa função faz o pivo tanto da linha quanto da coluna

        :param
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - ordem_matriz: a ordem da matriz.
           - pivo: um dos pivores da matriz_A.
           - pivo_linha: linha em que o pivo que está sendo passado como argumento se encontra.
           - pivo_coluna: coluna em que o pivo que está sendo passado como argumento se encontra.
           - vetor_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """
        if max(matriz_A[pivo_linha]) != pivo:
            max_coluna = np.argmax(matriz_A[pivo_linha])

            for i in range(ordem_matriz):
                matriz_A[i][pivo_coluna], matriz_A[i][max_coluna] = (
                    matriz_A[i][max_coluna],
                    matriz_A[i][pivo_coluna],
                )

            self.__colunas_invertidas.append(max_coluna)
            self.__colunas_invertidas.append(pivo_coluna)

        if (
            max(
                [
                    matriz_A[i][pivo_coluna]
                    for i in range(pivo_linha, ordem_matriz)
                ]
            )
            != pivo
        ):
            max_linha = max(
                range(pivo_linha, ordem_matriz),
                key=lambda i: matriz_A[i][pivo_coluna],
            )

            matriz_A[pivo_linha], matriz_A[max_linha] = (
                matriz_A[max_linha],
                matriz_A[pivo_linha],
            )
            vetor_b[pivo_linha], vetor_b[max_linha] = (
                vetor_b[max_linha],
                vetor_b[pivo_linha],
            )

    def get_resultados(self) -> List[float]:
        """
        Apenas retorna os resultados obtidos pelo método gauss-jacobi.
        Essa função é utilizada no diretórios de tests.

        :return - List[float]

        """
        return self.__resultados
