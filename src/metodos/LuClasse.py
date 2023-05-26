from typing import List


class MetodoLu:
    """
    Essa classe realiza todos os métodos necessáiros para resolver o sistema através
    do método de fatoração de LU.
    """

    def __init__(self):
        self.resultados_sistemas = []
        self.resultados_tempo = []
        self.copia_matriz_L = []
        self.copia_matriz_U = []
        self.linhas_trocadas = []

    def calcula_matriz_triangular_inferior_e_superior(
        self,
        ordem_matriz: int,
        matriz_A: List[List[float]],
        vetores_b: List[float],
        status_L_U: bool,
    ) -> None:
        """
        Essa função realiza calculos para obter uma matriz triangular inferior e passar essa matriz para  a funcao realiza_substituicao
        Essa função também chama a função realiza_privoteamento para diminui os erros de arredondamento e divisão por zero.
        Objetivo: Obter uma matriz triangular inferior e superior

        :param
           - ordem_matriz: a ordem da matriz.
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
           - status_L_U: valor booleano que indica se ja foi calculadora as matrizes L e U.
        :return - vazio

        """

        if status_L_U == True:
            self.aplica_pivoteamento_vetor_b(vetores_b)
            self.realiza_substituicoes(
                self.copia_matriz_L,
                self.copia_matriz_U,
                ordem_matriz,
                vetores_b,
            )
        else:
            lista_M = []
            matriz_l = [
                [0] * int(ordem_matriz) for _ in range(int(ordem_matriz))
            ]

            for etapas in range(0, ordem_matriz - 1):
                self.realiza_pivoteamento(
                    matriz_A, etapas, ordem_matriz, vetores_b
                )
                for linhas in range(etapas + 1, ordem_matriz):
                    lista_M.append(
                        matriz_A[linhas][etapas] / matriz_A[etapas][etapas]
                    )
                    matriz_l[linhas][etapas] = lista_M[0].copy()
                    for colunas in range(0, ordem_matriz):
                        matriz_A[linhas][colunas] = matriz_A[linhas][
                            colunas
                        ] - (lista_M[0] * matriz_A[etapas][colunas])
                    lista_M.clear()

            self.aplica_pivotemento_matriz_L(matriz_l)
            self.copia_matriz_L = matriz_l.copy()
            self.copia_matriz_U = matriz_A.copy()
            self.realiza_substituicoes(
                matriz_l, matriz_A, ordem_matriz, vetores_b
            )

    def realiza_substituicoes(
        self,
        matriz_L: List[List[float]],
        matriz_U: List[List[float]],
        ordem_matriz: int,
        vetores_b: List[float],
    ) -> None:
        """
        Essa função realiza o pivoteamento de uma matriz_A e dos vetores_b.Ela verifica se o pivo e o maior número da coluna
        Caso seja o maior, ela apenas retorna.
        Caso o pivo não seja o maior, ela troca as linhas.
        Objetivo: Minimizar os erros de arredondamento e a divisão por zero

        :param
           - matriz_L: matriz L que faz parte da equação matriz A = L * U.
           - matriz_U: matriz U que faz parte da equação matriz A = L * U.
           - ordem_matriz: a ordem da matriz do sistema.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """
        variaveis_resolvidas_Y = [0] * int(ordem_matriz)
        variaveis_resolvidas_X = [0] * int(ordem_matriz)

        variaveis_resolvidas_Y[0] = (vetores_b[0]) / (matriz_L[0][0])

        for i in range(1, ordem_matriz):
            soma = 0
            for j in range(i):
                soma += matriz_L[i][j] * variaveis_resolvidas_Y[j]
            variaveis_resolvidas_Y[i] = (vetores_b[i] - soma) / matriz_L[i][i]

        variaveis_resolvidas_X[ordem_matriz - 1] = (
            variaveis_resolvidas_Y[ordem_matriz - 1]
        ) / (matriz_U[ordem_matriz - 1][ordem_matriz - 1])

        for i in range(ordem_matriz - 2, -1, -1):
            soma = 0
            for j in range(i + 1, ordem_matriz):
                soma += matriz_U[i][j] * variaveis_resolvidas_X[j]
            variaveis_resolvidas_X[i] = (
                variaveis_resolvidas_Y[i] - soma
            ) / matriz_U[i][i]

        self.resultados_sistemas.append(variaveis_resolvidas_X)

    def realiza_pivoteamento(
        self,
        matriz_A: List[List[float]],
        etapas: int,
        ordem_matriz: int,
        vetores_b: List[float],
    ) -> None:
        """
        Essa função realiza o pivoteamento de uma matriz_A e dos vetores_b.Ela verifica se o pivo e o maior número da coluna
        Caso seja o maior, ela apenas retorna.
        Caso o pivo não seja o maior, ela troca as linhas e armazena as linhas trocadas
        Objetivo: Minimizar os erros de arredondamento e a divisão por zero

        :param
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - etapas: quantidade de etapas que deverao ser executadas, isso é decidido fazendo: ordem_matriz-1
           - ordem_matriz: a ordem da matriz do sistema.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """

        maior_numero = abs(matriz_A[etapas][etapas])
        pivo_atual = True
        linha_maior = 0

        for i in range(etapas + 1, ordem_matriz):
            if abs(matriz_A[i][etapas]) > maior_numero:
                maior_numero = abs(matriz_A[i][etapas])
                pivo_atual = False
                linha_maior = i

        if pivo_atual == False:
            matriz_A[linha_maior], matriz_A[etapas] = (
                matriz_A[etapas].copy(),
                matriz_A[linha_maior].copy(),
            )
            vetores_b[linha_maior], vetores_b[etapas] = (
                vetores_b[etapas].copy(),
                vetores_b[linha_maior].copy(),
            )

            self.linhas_trocadas.append(linha_maior)
            self.linhas_trocadas.append(etapas)
        else:
            return

    def aplica_pivoteamento_vetor_b(self, vetor_b: List[float]) -> None:
        """
        Essa função aplica o pivoteamento e corrigi o vetor_b caso exista mais de 2 sistemas
        no arquivo.

        :Param
            - vetor_b: São os vetores b adquiridos do arquivo.
        """

        for j in range(0, len(self.linhas_trocadas), 2):
            (
                vetor_b[self.linhas_trocadas[j]],
                vetor_b[self.linhas_trocadas[j + 1]],
            ) = (
                vetor_b[self.linhas_trocadas[j + 1]],
                vetor_b[self.linhas_trocadas[j]],
            )

    def aplica_pivotemento_matriz_L(self, matriz_L: List[List[float]]) -> None:
        """
        Essa função aplica o pivoteamento e corrigi a matriz L e seus elementos M.

        :Param
            - matriz_L: a matriz L representa a matriz da equação A = L * U
        """

        for j in range(0, len(self.linhas_trocadas), 2):
            if (
                self.linhas_trocadas[j] != 0
                and self.linhas_trocadas[j + 1] != 0
            ):
                (
                    matriz_L[self.linhas_trocadas[j]],
                    matriz_L[self.linhas_trocadas[j + 1]],
                ) = (
                    matriz_L[self.linhas_trocadas[j + 1]],
                    matriz_L[self.linhas_trocadas[j]],
                )

                matriz_L[self.linhas_trocadas[j]][
                    self.linhas_trocadas[j + 1]
                ] = matriz_L[self.linhas_trocadas[j + 1]][
                    self.linhas_trocadas[j + 1]
                ]

        for j in range(0, len(matriz_L)):
            matriz_L[j][j] = 1

    def add_resultados_tempo(self, tempo: int) -> None:
        """
        Essa função adiciona o tempo de execução a lista de resultados_tempo.
        Objetivo: adicionar tempo de execução

        :param
           - tempo: tempo gasto para executar cada sistema.
        :return - vazio

        """
        self.resultados_tempo.append(tempo)

    def imprime_resultados_lu(self, qtd_sistemas: int) -> None:
        """
        Essa função imprime o resultado dos sistemas que foram passados bem como o tempo de execução de cada sistema.
        É válido destacar que a primeira posicao do vetor resultados_sistemas representaria a variável X e assim por
        diante.
        Por exemplo, em um sistema 3x3 nos teriamos: variavel x = resultados_sistema[0], variavel y = resultados_sistemas[1]
        e por ultimo a variavel z = resultados_sistemas[2]
        Objetivo: imprimir resultados do sistema e tempo de execução.

        :param
           - qtd_sistemas: quantidade de sistemas no arquivo que foi fornececido.
        :return - vazio

        """

        print(
            "Resultados dos sistemas resolvidos pelo Metodo de fatoração de LU:"
        )
        print(
            f"Resultado para o sistema {qtd_sistemas+1}: {self.resultados_sistemas[qtd_sistemas]} Tempo: {self.resultados_tempo[qtd_sistemas]} segundos"
        )
        print()

    def get_resultados(self) -> List[float]:
        """
        Função usada para os testes dessa classe.

        :return - Uma lista contendo os resultados dos sitemas.

        """
        return self.resultados_sistemas
